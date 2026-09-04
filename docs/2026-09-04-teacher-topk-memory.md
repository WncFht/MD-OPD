# Teacher top-k 显存分析与优化

## 结论

本次 OOM 的主因是 `compute_top_k_teacher_log_probs` 中的完整
`logits.float()`。teacher 使用 BF16 forward，但 top-k 后处理把整个 response
vocab logits 转成 FP32；这个临时 tensor 单次就需要 32.46 GiB。此时 GPU
已经使用约 76.91 GiB，因此该申请不可能成功。

代码采用的主方案是：

1. 对 vocab logits 按 token 行分块后转 FP32。
2. 在同一个 FP32 chunk 内完成 `logsumexp`、`topk`、student candidate gather
   和 membership 判断，避免三次完整 FP32 转换。
3. 对显式支持 `logits_to_keep` 的 Transformers CausalLM，只让模型生成最后
   `response_length + 1` 个位置的 logits，从源头减少 prompt 部分的输出。
4. helper 直接接受 `(B,T,V)` response view，按每个 batch 的连续 token 行处理，
   避免旧调用中 `reshape(-1,V)` 因 stride 不连续而复制整份 BF16 response logits。

这些优化不改变 batch、序列长度、teacher micro batch、metric top-k 或任何
指标定义。

## 从日志复算实际尺寸

运行配置和 OOM 位置见：

`logs/0904-opd_math_full_DeepSeek-R1-Distill-Qwen-1.5B-T_1.0-2203.log`

关键配置为：

| 符号          | 含义                              |     值 |
| ------------- | --------------------------------- | -----: |
| `P`           | 最大 prompt 长度                  |   4096 |
| `R`           | 最大 response 长度                |   7168 |
| `S=P+R`       | teacher 输入序列长度              |  11264 |
| `B_cfg`       | `micro_batch_size_per_gpu` 配置值 |     24 |
| `B_local`     | 每个 DP rank 实际 forward 行数    |      8 |
| `V`           | teacher vocab size                | 151936 |
| `K_s` / `K_t` | student / teacher top-k           |     16 |
| dtype         | teacher logits                    |   BF16 |

日志中失败申请为 32.46 GiB。用 response token 数计算：

```text
N = B_local * R = 8 * 7168 = 57344
N * V * sizeof(float32)
= 57344 * 151936 * 4 bytes
= 32.457 GiB
```

这与日志的 `Tried to allocate 32.46 GiB` 完全吻合，因此失败申请就是
`logits.float()` 的 response 副本。`B_cfg=24` 是每个 teacher micro-batch 的
上限；本次 64 条训练 batch 分到 8 个 DP rank，每个 rank 实际拿到 8 条，故
OOM 计算使用 `B_local=8`。

## 当前实现的显存账

以下只列与 top-k 相关的主要 tensor，单位为 GiB，按二进制 GiB 计算：

| tensor / 临时量                               | 形状                           |                       显存 |
| --------------------------------------------- | ------------------------------ | -------------------------: |
| full `output.logits`                          | `(8, 11264, 151936)`, BF16     |                     25.502 |
| response slice                                | `(8, 7168, 151936)`, BF16 view |      0（共享上项 storage） |
| `reshape(-1,V)`（旧 top-k / CE 路径可能触发） | `(57344, 151936)`, BF16        |                     16.229 |
| `logits.float()` for denominator              | `(57344, 151936)`, FP32        |                 **32.457** |
| `logits.float()` for top-k                    | 同上                           | 32.457（另一时刻的临时量） |
| `logits.float()` for student gather           | 同上                           | 32.457（另一时刻的临时量） |
| teacher top-k logits                          | `(57344, 16)`, FP32            |                     0.0034 |
| teacher top-k ids                             | `(57344, 16)`, int64           |                     0.0068 |
| `matches`                                     | `(57344, 16, 16)`, bool        |                     0.0137 |
| `log_denom`                                   | `(57344, 1)`, FP32             |                     0.0002 |

旧版 sampled-token `logprobs_from_logits` 也会对非连续 response view 做一次
`reshape(-1,V)`；这份约 16.23 GiB 的 BF16 copy 通常在 CE 调用结束后释放，但
会抬高中间峰值。新代码按 batch row 调用 CE，每个 row 的 `(T,V)` view 连续，
因此不再需要该整块 copy。

因此 `matches`、top-k 输出和 metric 汇总 tensor 都不是这次 OOM 的主因。日志
显示：

```text
GPU total:       79.25 GiB
free:             1.72 GiB
PyTorch allocated:75.96 GiB
non-PyTorch use: 约 0.95 GiB
```

申请 32.46 GiB 后理论使用量超过 108 GiB，远大于 GPU 容量；这不是 allocator
碎片问题（reserved but unallocated 只有 43.65 MiB）。

## 优化后的峰值估算

实现将 FP32 vocab 临时量限制为最多 512 MiB，并且最多处理 1024 行。对本次
`V=151936`：

```text
rows_by_bytes = floor(512 MiB / (151936 * 4)) = 883
chunk_rows = min(1024, 883) = 883
FP32 chunk = 883 * 151936 * 4 bytes = 0.505 GiB
```

实际峰值还包括 top-k workspace 和 `matches`，但仍约为原来 32.46 GiB 临时
量的 1/64。即使不支持 `logits_to_keep`，且不考虑旧 `reshape` 的额外复制，原日志的 75.96 GiB 基线只需再加约
0.5 GiB；支持时还可把 full output 从 25.50 GiB 降到约 16.23 GiB：

```text
response-only output = 8 * (7168 + 1) * 151936 * 2 bytes
                     ≈ 16.23 GiB
prompt logits saved   ≈ 25.50 - 16.23 = 9.27 GiB
```

`logits_to_keep` 只改变模型输出中保留的 position 范围；保留下来的最后
`R+1` 个位置仍通过原来的 `[:, -R-1:-1, :]` 取 response logits，位置和数值
语义不变。不声明该参数的模型继续走兼容路径。

## 数值与指标语义

分块只改变处理顺序，不改变每一行使用 FP32 `logsumexp` 的事实；top-k 在同一
个 FP32 chunk 上执行，排序输入与旧实现的 `logits.float()` 完全相同。

必须保持以下现有语义：

- `top_k_overlap_ratio`：response 有效 token 中，student top-k 命中 teacher
  top-k 的 candidate 数，占 `有效 token 数 * K_s`。
- `top_k_student_mass`：student top-k log-prob 的 `exp().sum(-1)` 的有效
  token 平均。
- `top_k_teacher_mass`：teacher 在 **student candidate 集合** 上的 log-prob
  质量，不是 teacher 自己 top-k 集合的质量。
- `top_k_strategy=only_stu` 下 reward 仍使用 student candidate 集合；metric-only
  模式下 `rm_scores` 仍是普通逐 token reverse KL。
- `response_mask` 的应用位置和 OPD trainer 中的 DP metric 汇总逻辑不变。

## 计算量与复杂度权衡

| 项目         | 影响                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| 数值精度     | 保留旧的 FP32 denominator 和 top-k 输入；不改成 BF16 `logsumexp`                                        |
| 显存         | vocab 临时量从 `O(NV)` 降到 `O(chunk_rows·V)`；支持 `logits_to_keep` 时额外减少 prompt logits           |
| 计算量       | 增加约 `B_local * ceil(R/chunk_rows)` 次后处理 kernel 调度；本次约 72 个 chunk，远小于一次 OOM 重试代价 |
| forward 计算 | `logits_to_keep` 支持时减少 lm-head 输出写回；Transformer 主体计算不变                                  |
| 实现复杂度   | 只改 teacher top-k helper 和兼容 forward kwargs；不改 reward / metric 定义                              |

没有采用直接降低 batch、序列长度、`TEACHER_MICRO_BSZ`、`metric_top_k` 或
切换到 BF16 denominator，因为这些要么改变运行参数，要么改变数值语义。
