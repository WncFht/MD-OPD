# OPD 训练全流程显存账

本文根据以下运行日志和当前代码整理：

```text
logs/0904-opd_math_full_DeepSeek-R1-Distill-Qwen-1.5B-T_1.0-2340.log
```

目标是说明一个 training step 中显存如何被申请、复用和释放，并给出可以
代入其他 batch、序列长度、词表和并行度的参数化公式。本文使用二进制 GiB：

```text
1 GiB = 2^30 bytes
M(shape, dtype) = product(shape) * sizeof(dtype) / 2^30
```

## 1. 运行参数与符号

| 符号            | 含义                                      | 本次值 / 来源                                                        |
| --------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| `G`             | 单卡 GPU 总显存                           | `79.25 GiB`                                                          |
| `D`             | actor/teacher 数据并行 rank 数            | `8`                                                                  |
| `B`             | training batch size                       | `64`                                                                 |
| `B_d`           | 每个 DP rank 的样本数                     | `B / D = 8`                                                          |
| `P`             | 最大 prompt 长度                          | `4096`                                                               |
| `R`             | 最大 response 长度                        | `7168`                                                               |
| `L`             | 最大完整序列长度                          | `P + R = 11264`                                                      |
| `V`             | teacher/student vocab size                | `151936`，由上一份 OOM 的申请尺寸精确反推                            |
| `K_s`, `K_t`    | student / teacher top-k                   | `16`, `16`                                                           |
| `N_p^A`         | actor Qwen2 参数量                        | `1.78B`，日志明确打印                                                |
| `N_p^T`         | teacher 参数量                            | 未在本日志逐独打印，公式保留为变量；若 teacher 同为 1.78B 可直接代入 |
| `T_cap`         | actor 每卡 dynamic token 上限             | `32768`                                                              |
| `B_teacher_cfg` | teacher `micro_batch_size_per_gpu`        | `24`                                                                 |
| `b_logit`       | forward logits dtype 字节数               | BF16 = `2`                                                           |
| `b_fp32`        | 数值稳定后处理 dtype 字节数               | FP32 = `4`                                                           |
| `b_param`       | actor 持久参数 dtype 字节数               | FP32 = `4`                                                           |
| `b_act`         | actor forward logits/softmax dtype 字节数 | 通常 BF16 = `2`（由 autocast 决定）                                  |

本次使用：

- `use_remove_padding=True`；
- actor `use_dynamic_bsz=True`，`ppo_max_token_len_per_gpu=32768`；
- actor `model_dtype=fp32`，FSDP mixed-precision forward 为 BF16；
- actor `param_offload=True`、`optimizer_offload=True`；
- teacher dtype 为 BF16，FSDP `CPUOffload(offload_params=True)`；
- `metric_top_k=16`、`log_prob_top_k=0`，所以 top-k 只用于指标，不替换普通
  reverse-KL reward；
- `adv_estimator=token_reward_direct`，日志第 1、2 行明确显示 critic 被禁用。

## 2. 先解释日志中的 token 数

`_balance_batch` 计算每条样本的完整有效序列长度，然后将 64 条样本重排到
8 个 DP rank。`log_seqlen_unbalance` 的 `global_seqlen/*` 是 **每个 DP
partition 的 token 总数**，不是一条样本的长度：

```text
global_seqlen/mean = total_valid_tokens / D
```

日志中的三步可以精确复算：

| step | `perf/total_num_tokens` |     `/ D` | 日志 `global_seqlen/mean` | `/ B`（每条样本平均） |
| ---: | ----------------------: | --------: | ------------------------: | --------------------: |
|    1 |                  420123 | 52515.375 |                 52515.375 |           6564.421875 |
|    2 |                  412634 |  51579.25 |                  51579.25 |            6447.40625 |
|    3 |                  409214 |  51151.75 |                  51151.75 |            6393.96875 |

step 1 还满足：

```text
mean(prompt_length) + mean(response_length)
= 93.46875 + 6470.953125
= 6564.421875
```

这说明 token 统计、batch size 和 DP 聚合是一致的。`balanced_min/max` 是
重排后各 rank 的 token 总数；step 1 为 `52170..52680`，所以 actor 每卡约
有 52k token，随后由 dynamic batching 拆成至少两个 `T_cap=32768` 的
forward micro-batch：

```text
N_micro_actor,d <= ceil(G_rank / T_cap)
```

本次 `G_rank≈51k..53k`，因此通常是 2 个；实际每个 micro-batch 的 token
数由 sequence balancing 决定，不一定恰好等于 32768。

## 3. GPU、进程和 allocator 的三个口径

必须区分：

1. `torch.cuda.memory_allocated()`：当前 PyTorch tensor 实际占用；
2. `torch.cuda.memory_reserved()`：PyTorch caching allocator 向 CUDA 保留的
   总量，包含尚未重新分配的缓存；
3. `device memory used` / `nvidia-smi`：包括 CUDA driver、通信库、vLLM
   子进程和其他非 PyTorch 分配。

本仓库还将 actor 和 OPD teacher 注册到同一个 resource pool，并由
`create_colocated_worker_cls` 在一个 `WorkerDict` Ray 进程内实例化。因此每张
GPU 上通常有一个同时持有 actor/teacher Python 对象的 WorkerDict，以及独立的
vLLM server/worker 进程；但训练 step 中这些模型的重型 forward 是串行的：
rollout 后先 sleep vLLM，old-log-prob 结束后 actor offload，随后才进入 teacher
score。对象共存不等于 full logits 共存。

日志的初始化采样为：

```text
Before FSDP: allocated=0.00 GiB, reserved=0.00 GiB,
             device used=0.41 / 79.25 GiB
After FSDP:  allocated=0.83 GiB, reserved=6.26 GiB,
             device used=7.15 / 79.25 GiB
```

训练 step 中 actor 输出的高水位为：

```text
actor/perf/max_memory_allocated_gb = 50.24179744720459
actor/perf/max_memory_reserved_gb  = 50.8671875
```

这两个指标在 step 1、2、3 完全相同，是 worker 进程从启动以来的 high-water
mark；代码在 `_postprocess_output` 读取 `max_memory_*`，但没有在每个阶段前
调用 `reset_peak_memory_stats()`。因此它不是“当前 teacher 阶段占用”，也不能
据此把 actor、teacher、vLLM 三者相加。当前 log 没有 per-phase teacher GPU
采样，所以 teacher 阶段只能用公式给出理论峰值。

## 4. 初始化阶段

### 4.1 Actor FSDP 参数

日志打印 actor 为 `N_p^A=1.78B` 参数，训练 `model_dtype=fp32`。未分片的完整
FP32 参数理论大小为：

```text
M_actor_full_fp32 = N_p^A * b_param
                  = 1.78e9 * 4 / 2^30
                  ≈ 6.631 GiB
```

8-way FSDP shard 的理论大小为：

```text
M_actor_shard_fp32 = N_p^A * b_param / D
                   ≈ 0.829 GiB
```

它与日志的 `After FSDP allocated=0.83 GiB` 对齐。`reserved=6.26 GiB` 不是
参数本身，而是 allocator 保留、FSDP flat-param/metadata、CUDA runtime 和
初始化阶段临时量的合计。

如果只看 BF16 forward 参数，完整模型和一个理想静态 shard 分别为：

```text
M_actor_full_bf16  = N_p^A * 2 / 2^30     ≈ 3.316 GiB
M_actor_shard_bf16 = N_p^A * 2 / D / 2^30 ≈ 0.414 GiB
```

但本次 actor 的持久参数 storage 仍按 FP32 计算；BF16 主要出现在 FSDP
forward 的参数/activation 计算路径。

### 4.2 Actor optimizer

AdamW 的两个 moment 通常是 FP32。若 moment 在 GPU，理论 shard 为：

```text
M_adam_gpu = 2 * N_p^A * b_param / D / 2^30
           ≈ 1.658 GiB
```

本次 `optimizer_offload=True`，所以这部分应驻留 CPU，而不是长期占用 GPU。
首次 `optimizer.step()` 后 optimizer state 才会完整建立；它仍影响 host memory，
但不应被误算进 actor 的 GPU 参数 shard。

### 4.3 Teacher FSDP 参数

teacher 使用 BF16 和 `CPUOffload(offload_params=True)`。因此稳定状态是：

```text
GPU steady teacher params ≈ 0
CPU teacher params / rank ≈ N_p^T * 2 / D
CPU teacher params / cluster ≈ N_p^T * 2
```

forward 时 FSDP 仍需逐层 all-gather 当前 flat-param；其峰值取决于 wrap policy
和单个 wrapped handle 的大小，不能用整模型 `N_p^T*2` 直接当成 GPU 常驻量。

### 4.4 vLLM 子进程

vLLM 是独立进程，使用 `bfloat16`、`tensor_parallel_size=1`、
`gpu_memory_utilization=0.75`、`max_model_len=max_num_batched_tokens=35840`。
`0.75` 是 vLLM 的预算上限，不是实际已分配量。它与 actor/teacher 的 CUDA
allocator 统计分开；同卡上的最终 device used 还包括它的权重、KV cache、IPC
和 CUDA runtime。

KV cache 的参数化大小为：

```text
M_KV = N_KV_tokens * 2(K,V) * n_layers * n_kv_heads * head_dim * b_kv / 2^30
```

其中 `N_KV_tokens` 由 vLLM scheduler 实际占用，最多受
`max_num_batched_tokens`、`max_model_len`、`max_num_seqs` 和 GPU budget 共同
限制。当前代码在 rollout 结束后调用 `checkpoint_manager.sleep_replicas()`；
vLLM hybrid mode 默认用 level 2 sleep（无 MTP/LoRA 特殊情况），会释放权重和
KV cache，使 actor/teacher 阶段可用显存显著增加。

作为数量级校验，若该 checkpoint 的 config 采用常见 Qwen2 1.5B 结构
`n_layers=28, n_kv_heads=2, head_dim=128`，且 KV 使用 BF16，则：

```text
bytes_per_KV_token = 2(K,V) * 28 * 2 * 128 * 2 = 28672 bytes
M_KV(35840 tokens)  = 35840 * 28672 / 2^30 ≈ 0.957 GiB
```

这只是基于 architecture config 的条件示例；实际 vLLM block metadata、
allocator 和并发 request 会在此基础上增加开销，最终仍由 `0.75` budget 控制。

可将 vLLM 侧约束写成：

```text
M_vllm_weights + M_KV + M_vllm_workspace
<= u * M_vllm_available,   u = gpu_memory_utilization = 0.75
```

`M_vllm_available` 是 vLLM 初始化时看到的可用显存，不等于整张卡的
`79.25 GiB`；因此不能直接用 `0.75 * 79.25` 作为实际 KV 大小。

## 5. Rollout 阶段

rollout 阶段主要由 vLLM 子进程持有：

```text
prompt -> vLLM prefill/decode -> response + rollout log-probs -> CPU/Ray batch
```

训练主进程只保留生成后的 token、mask 和可选 rollout log-probs；这些数据量按
`O(B*R)`，相对 vocab logits 很小。生成结束后立即 sleep replicas，因此 rollout
KV cache 不应与后续 actor/teacher full logits 峰值同时存在。

日志中的生成时间：

| step | `timing_s/gen` | `generate_sequences/mean` |
| ---: | -------------: | ------------------------: |
|    1 |      159.112 s |                 123.052 s |
|    2 |      128.873 s |                 110.558 s |
|    3 |      127.099 s |                 109.275 s |

这些是时间，不是显存采样；不能由时间反推出 KV 的确切大小。

## 6. Actor old-log-prob 阶段

调用链为：

```text
_compute_old_log_prob
  -> actor_rollout_wg.compute_log_prob
  -> FSDP infer_batch(forward_only=True)
  -> prepare_model_outputs
```

本阶段的 actor 模型需要暂时加载到 GPU；rollout replicas 已在上一步 sleep。
`use_dynamic_bsz=True` 时每个 actor micro-batch 的 **packed 完整序列** token 数
记为 `T_A`（包含 prompt 和 response；不是只计算 response），满足：

```text
T_A <= T_cap = 32768
```

### 6.1 原始 logits

actor 使用 eager/non-fused logits 路径，单个 micro-batch 的 raw logits 约为：

```text
M_logits_bf16(T_A) = T_A * V * 2 / 2^30
```

取 token 上限 `T_A=32768`：

```text
M_logits_bf16 = 32768 * 151936 * 2 / 2^30
              ≈ 9.273 GiB
```

### 6.2 entropy 临时量

`_compute_old_log_prob` 显式传 `calculate_entropy=True`。当前
`entropy_from_logits` 先执行 `softmax(logits)`，且没有 chunking，因此可能同时
存在一份同形状的概率 tensor：

```text
M_entropy_softmax(T_A) ≈ T_A * V * b_act / 2^30
                       ≈ 9.273 GiB  (b_act=2 的 BF16 假设)
```

entropy 输出本身只有 `T_A*4` 字节，几乎可以忽略；真正占空间的是 softmax
workspace。

### 6.3 Student top-k 的 FP32 workspace

metric-only top-k 仍要求 actor 提取 student support set。当前
`_emit_opd_student_top_k` 使用：

```python
torch.logsumexp(logits.float(), dim=-1, keepdim=True)
```

所以会把整个 `(T_A,V)` logits 转为 FP32：

```text
M_student_fp32(T_A) = T_A * V * 4 / 2^30
```

在 `T_A=32768` 时：

```text
M_student_fp32 = 32768 * 151936 * 4 / 2^30
                ≈ 18.547 GiB
```

这两个 workspace 在 `prepare_model_outputs` 中是顺序执行的，并非同时存活。
因此 old-log-prob 的 vocab 相关峰值应取两种阶段的最大值：

```text
entropy phase: 9.273 (logits) + 9.273 (softmax) = 18.547 GiB
top-k phase:   9.273 (logits) + 18.547 (FP32 copy) = 27.820 GiB
vocab peak = max(18.547, 27.820) = 27.820 GiB
```

日志 actor high-water mark 为 `50.242 GiB`，相对于上述 vocab 相关峰值还剩约
`22.422 GiB` 的空间可归因于 Transformer hidden states、
gradient-checkpointing/recompute workspace、FSDP 参数 all-gather、
attention/communication workspace 和 allocator cache：

```text
50.242 - 27.820 ≈ 22.422 GiB
```

这是“顺序 workspace 峰值 + 实现 workspace”意义上的吻合，不是逐 tensor profiler 的精确
分解。当前日志没有 reset peak，也没有在 old-log-prob 前后打印 allocated，因而
无法从日志单独证明 50.242 GiB 究竟发生在 old-log-prob 还是 actor update；但
上述公式解释了为何该 high-water mark 落在 50 GiB 量级。若 kernel 在某阶段
额外保留 logits 或使用 FP32 softmax，实际 workspace 会高于这些裸 tensor 公式。

### 6.4 本阶段释放顺序

1. `prepare_model_outputs` 完成后，raw logits、softmax 和 FP32 top-k workspace
   的引用消失；
2. `log_probs`、`entropy`、student top-k ids/log-probs 转为结果 tensor；
3. `infer_batch` 的 `EngineEvalModeCtx.__exit__` 重新 reshard，并按
   `param_offload=True` 将 actor FSDP 参数搬回 CPU；
4. driver 计算 entropy metric 后立即 `old_log_prob.batch.pop("entropys")`；
5. `batch.union(old_log_prob)` 只保留 `old_log_probs` 和 top-k 结果，均为
   `O(B*R)` 或 `O(B*R*K)`，不再保留 vocab logits。

## 7. Teacher score 阶段

调用链为：

```text
compute_opd_token_reward
  -> opd_teacher_wg.compute_rm_score
  -> TeacherWorker._forward_micro_batch
  -> compute_top_k_teacher_log_probs
```

此时 actor 已从 eval context offload，vLLM 也处于 sleep；teacher 是当前主要
GPU 计算者。teacher 配置 `micro_batch_size_per_gpu=24` 是上限，实际 DP rank
样本数为 `B_d=8`，因此本次最大 response token 数为：

```text
N_T = B_d * R = 8 * 7168 = 57344
```

### 7.1 如果不做 output slicing

完整输入 logits（prompt + response）为：

```text
M_teacher_full_bf16 = B_d * (P+R) * V * 2 / 2^30
                    = 8 * 11264 * 151936 * 2 / 2^30
                    ≈ 25.502 GiB
```

response slice 是同一 storage 的 view，不额外占用；如果后续对非连续 view
执行 `reshape(-1,V)`，还可能产生：

```text
M_response_copy_bf16 = B_d * R * V * 2 / 2^30
                     ≈ 16.229 GiB
```

如果底层 CausalLM 支持当前 teacher worker 的 `logits_to_keep=R+1`，模型只
保留用于 response 对齐的最后 `R+1` 个 position：

```text
M_teacher_keep_bf16 = B_d * (R+1) * V * 2 / 2^30
                    ≈ 16.231 GiB
```

这会把 prompt 部分原本约 `B_d*P*V*2/2^30≈9.273 GiB` 的 output storage
直接省掉；是否实际触发取决于模型 forward signature，当前日志没有打印该
能力检测结果。

### 7.2 当前 top-k FP32 分块上限

当前实现将 vocab 后处理的 FP32 临时量限制在：

```text
C = 512 MiB
q = min(1024, floor(C * 2^30 / (V * 4)))
```

对 `V=151936`：

```text
q = min(1024, floor(512*2^20 / (151936*4)))
  = 883 rows
M_chunk_fp32 = 883 * 151936 * 4 / 2^30
             ≈ 0.500 GiB
```

旧版 OOM 的失败申请为：

```text
N_T * V * 4 / 2^30
= 57344 * 151936 * 4 / 2^30
= 32.457 GiB
```

这正是上一份日志的 `Tried to allocate 32.46 GiB`。新实现把这项从
`O(N_T*V)` 降为 `O(q*V)`；因此即使不依赖 `logits_to_keep`，单次 FP32 vocab
workspace 也只有约 `0.500 GiB`。

### 7.3 top-k 输出和指标 tensor

最大 response 长度下，单个 DP rank 的主要输出为：

```text
M_teacher_candidate_lp = B_d * R * K_t * 4 / 2^30 ≈ 0.0034 GiB
M_teacher_candidate_id = B_d * R * K_t * 8 / 2^30 ≈ 0.0068 GiB
M_matches              = B_d * R * K_s * K_t * 1 / 2^30 ≈ 0.0137 GiB
M_teacher_log_probs    = B_d * R * 4 / 2^30 ≈ 0.0017 GiB
```

这些量相比 32.457 GiB 失败申请都很小。`compute_rm_score` 结束时
`DataProto.to("cpu")` 将结果搬到 CPU；teacher FSDP CPU offload 也使参数不在
GPU 常驻。日志中 `timing_s/opd_teacher` 为：

```text
step 1: 5.372 s
step 2: 3.674 s
step 3: 6.040 s
```

说明修复后的 teacher forward/top-k 已实际完成，但日志没有记录该阶段的
`max_memory_allocated`，所以不能把 `0.500 GiB` 误报为实测峰值；它是 vocab
后处理临时量的理论上限。

### 7.4 本阶段释放顺序

1. 每个 teacher micro-batch 的 BF16 output logits 只活到 `_forward_micro_batch`
   返回；
2. FP32 chunk 在该 chunk 写回结果后释放，下一 chunk 复用 allocator；
3. top-k 结果只保留 `O(B_d*R*K)` tensor；
4. `compute_rm_score` 返回前将 output 搬 CPU；
5. `rm_scores`、`teacher_log_probs` 等最终 batch tensor 只占 `O(B*R)`，不再有
   vocab 维度。

## 8. Reward、advantage 和 actor update

### 8.1 Reward / OPD token reward

当前是 metric-only top-k：

```text
rm_scores = -(old_log_probs - teacher_log_probs)
```

`compute_opd_token_reward` 将原 verifier tensor 保存到
`true_token_level_scores`，再把 `rm_scores` 写入 `token_level_scores`。这些都
是 `(B,R)` FP32：

```text
M_(B,R,fp32) = B * R * 4 / 2^30
             = 64 * 7168 * 4 / 2^30
             ≈ 0.0017 GiB
```

即使同时保留 `old_log_probs`、`teacher_log_probs`、`rm_scores`、
`token_level_rewards`、`advantages`、`returns` 等十个同形状 tensor，也只有
约 `0.017 GiB`。top-k 的 `(B,R,K)` FP32 tensor 约 `0.0273 GiB`，ids 约
`0.0547 GiB`。因此 driver/batch tensor 不是 GPU OOM 主因。

更完整地，最大 padding 形状下常见 batch tensor 的大小为：

```text
input_ids / position_ids (int64) = B * L * 8 / 2^30
attention_mask / response_mask  = B * L * 1 / 2^30
one (B,R) FP32 field            = B * R * 4 / 2^30
student_top_k_log_probs         = B * R * K_s * 4 / 2^30
student_top_k_ids                = B * R * K_s * 8 / 2^30
```

代入 `B=64, L=11264, R=7168, K_s=16`，单个 `input_ids` 约 `0.0054 GiB`，
单个 `(B,R)` FP32 field 约 `0.0017 GiB`。这些对象在 worker 返回时通常已经
`.cpu()`，即使短暂位于 GPU 也远小于 vocab logits。

### 8.2 Critic 被禁用

日志开头：

```text
Disabled critic as algorithm.adv_estimator != gae
```

所以本次没有真正的 critic value forward、critic backward 或 critic optimizer
state GPU 峰值。日志中的 `critic/score/*`、`critic/rewards/*` 是 reward/score
统计键名，不代表 critic 模型被执行。

同理，日志配置中的 `algorithm.use_kl_in_reward=False` 且 actor
`use_kl_loss=False`，因此 `need_reference_policy()` 返回 false；本次也没有
reference-policy forward 或 reference-model 显存。teacher reverse-KL 使用的
`old_log_probs` 和 `teacher_log_probs` 是已保存的 `(B,R)` 结果，不需要第三个
reference 模型。

### 8.3 Actor update

`_update_actor` 再次以 dynamic token cap `T_A<=32768` 调 actor FSDP。与
old-log-prob 不同，训练路径需要 autograd：

```text
M_update(T_A) = M_param_shard
              + M_grad_shard
              + M_FSDP_allgather
              + M_hidden_saved(T_A, checkpointing)
              + M_logits/CE_workspace(T_A)
              + M_optimizer_gpu
```

如果模型 hidden size 为 `H`、Transformer 层数为 `L_layer`、checkpoint 保留
`L_ckpt` 个边界，activation 的一阶估算可以写成：

```text
M_hidden_saved(T_A) ≈ c_hidden * T_A * H * L_ckpt * b_act / 2^30
M_attention_workspace(T_A) ≈ c_attn * T_A * H * b_act / 2^30
```

`c_hidden`、`c_attn` 由具体 Qwen block、FlashAttention 版本、是否保存 residual
和 kernel workspace 决定；它们不能从当前日志单独识别，所以本文不把这部分
伪装成精确常数。没有 gradient checkpointing 时，`L_ckpt` 近似为
`L_layer`；开启 checkpointing 后通常只保留每个 checkpoint segment 的边界。

以 `H=1536`、`T_A=32768`、`b_act=2` 为例，单个 hidden 向量的基数为：

```text
T_A * H * b_act / 2^30
= 32768 * 1536 * 2 / 2^30
≈ 0.09375 GiB
```

实际保存多少份由 `c_hidden * L_ckpt` 决定，这也是为什么不能仅凭参数量把
actor update activation 精确换算成一个固定 GiB。

其中：

- `M_param_shard≈N_p^A*4/D≈0.829 GiB` 是日志已验证的 FP32 shard 量级；
- `M_optimizer_gpu` 的长期驻留为 0，因为 optimizer offload 开启；但 actor
  update 的外层 `train_mode` 会暂时把 Adam state 加载到 GPU，若按 FP32 moment
  估算，其瞬时 shard 约为 `2*N_p^A*4/D≈1.658 GiB`，随后在 context exit
  搬回 CPU；
- `M_hidden_saved` 受 Transformer 层数、hidden size、gradient checkpointing
  和 attention backend 影响，不能只由当前日志精确反推；
- `train_mini_batch` 的外层 train context 负责一次性加载 model/optimizer/grad，
  内层每个 micro-batch 设置 `disable_auto_offload=True`，避免在 micro-batch
  之间反复搬运；
- `forward_backward_batch` 每个 micro-batch backward 后默认丢弃
  `model_output`，不会把所有 micro-batch 的 logits/activation 累积到整个
  mini-batch；
- `EngineTrainModeCtx.__exit__` 会 zero-grad，并按 offload 配置切回 CPU。

因此 actor update 的峰值随 `T_A` 变化，而不是随完整 `B=64` 一次性变化。日志
记录的 `actor/perf/max_memory_allocated_gb=50.242` 是该 worker 生命周期内的
最高点；当前日志无法把它精确归因到 old-log-prob 或 update，但第 6 节的
`27.820 GiB` vocab 相关峰值已经说明了它的数量级来源。

### 8.4 Weight sync 与下一轮 rollout

actor update 完成后，checkpoint manager 将 actor 权重发送给 vLLM：

```text
actor FSDP model/optimizer/grad offload
-> vLLM weight update
-> clear KV cache
-> resume KV cache / weights
```

若 checkpoint backend 使用双 buffer，权重同步额外 buffer 上界可写为：

```text
M_sync_buffer <= n_buffers * U
```

其中 `U` 是 `update_weights_bucket_megabytes * 2^20`，`n_buffers` 由 backend
决定（常见为 1 或 2）。当前日志没有打印 bucket 大小和同步时的 memory sample，
因此该项只能保留参数化表达式，不能给出实测 GiB。

## 9. 一个 step 的生命周期图

```text
初始化
  actor FSDP shard + CPU Adam states
  teacher CPU-offloaded FSDP
  vLLM server ready
       |
       v
rollout (vLLM weights + KV cache active)
       |
       +--> sleep_replicas(): release rollout weights/KV
       v
actor old_log_prob (actor model + logits + entropy + student top-k FP32)
       |
       +--> eval exit: reshard/offload actor; logits/workspace released
       v
teacher score (teacher output + <=0.500 GiB FP32 chunk)
       |
       +--> DataProto.to("cpu"); teacher output/workspace released
       v
reward / advantage (主要是 O(B*R) CPU batch tensors; critic disabled)
       v
actor update (autograd + checkpointed activations, T_A <= 32768)
       |
       +--> backward 后丢弃 model_output; offload model/optimizer/grad
       v
weight sync -> vLLM wake -> 下一轮 rollout
```

关键点是：actor old-log-prob、teacher score、actor update 在代码流程上是串行
阶段，不能把它们各自的 vocab logits 峰值直接相加。真正需要防范的是同一阶段
内的 live tensor + 临时 workspace + 外部进程显存。

## 10. 理论值与日志值对照

| 项目                         | 理论式 / 理论值                         | 日志值                       | 结论                   |
| ---------------------------- | --------------------------------------- | ---------------------------- | ---------------------- |
| Actor 参数 shard             | `1.78B*4/8 ≈ 0.829 GiB`                 | `After FSDP allocated=0.83`  | 精确对齐               |
| step 1 global mean           | `420123/8=52515.375`                    | `52515.375`                  | 精确对齐               |
| step 2 global mean           | `412634/8=51579.25`                     | `51579.25`                   | 精确对齐               |
| step 3 global mean           | `409214/8=51151.75`                     | `51151.75`                   | 精确对齐               |
| Actor old-log BF16 logits    | `32768*151936*2/2^30=9.273`             | 未逐阶段打印                 | 理论项                 |
| Actor student FP32 workspace | `32768*151936*4/2^30=18.547`            | 未逐阶段打印                 | 理论项，直接对应代码   |
| Actor vocab 相关峰值         | `max(9.273+9.273, 9.273+18.547)=27.820` | high-water `50.242`          | 剩余为模型/workspace   |
| 旧 Teacher OOM申请           | `57344*151936*4/2^30=32.457`            | 上一份日志 `32.46 GiB`       | 精确对齐               |
| 新 Teacher FP32 chunk        | `883*151936*4/2^30≈0.500`               | 新日志无 teacher peak sample | 理论上限，不冒充实测   |
| Allocator slack              | `reserved-allocated`                    | `0.625 GiB`                  | 正常缓存，不是主要碎片 |

## 11. 结论与边界

1. 新日志已经连续完成 3 个 step；没有出现 teacher OOM，说明分块后处理解除了
   上一份日志中确定性的 32.46 GiB 单次申请。
2. 这次日志中最可精确解释的 GPU 峰值是 actor worker 的 `50.242 GiB`。其
   vocab 相关峰值是 `max(9.273+9.273, 9.273+18.547)=27.820 GiB`；entropy
   softmax 与 student top-k FP32 workspace 不同时存活，剩余部分属于模型
   activation、FSDP 和 backend workspace。
3. teacher 的新峰值不能从当前日志直接读取；可以确定的理论上限是 `0.500 GiB`
   FP32 vocab chunk，另加 BF16 output、FSDP all-gather、attention workspace 和
   top-k 小 tensor。
4. `perf/max_memory_*` 是进程生命周期 high-water mark，必须配合阶段前后的
   `reset_peak_memory_stats()` 或 explicit memory samples 才能做精确归因。
5. host memory 日志的 `106.18/123.62/123.80 GiB` 是整机 CPU virtual memory
   used，不是 GPU memory，也不是某个单独 tensor 的大小；它不能与 79.25 GiB
   GPU 容量直接比较。

## 12. 如何把理论变成逐阶段实测

若要把本文的参数化账进一步收敛为逐 tensor 的实测账，建议在每个阶段边界：

```text
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
log allocated / reserved / device used
执行一个阶段
torch.cuda.synchronize()
log allocated / reserved / max_allocated / max_reserved
```

同时记录 `T_A`、`output.logits.shape`、dtype、`is_contiguous` 和
`N_KV_tokens`。actor/teacher 的 PyTorch stats 应按 WorkerDict 进程分别记录；
vLLM 则需要在其 server/worker 进程中记录，不能从 actor worker 的
`max_memory_allocated` 推断。这样可以将本文中的
`M_hidden_saved`、`M_FSDP_allgather`、`M_vllm_workspace` 三个参数化余项拆成
实际数值。
