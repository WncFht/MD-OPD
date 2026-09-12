# Token 4-gram Jaccard Diversity

按 arXiv:2608.31046(OPSA)Appendix B.3 的定义，计算各模型在各 bench 上
`justrl_eval_outputs` 评测 rollout 的 response 多样性 D_J(L)。

> 训练内监控：同一族指标的在线版已接入 verl validation
> ([`verl/verl/trainer/ppo/diversity_metrics.py`](../verl/verl/trainer/ppo/diversity_metrics.py)),
> 按"相对位置网格 + 绝对锚点"动态取点，输出 `val-aux/<bench>/jaccard/...` 与
> `val-aux/<bench>/answer/...`,配置见 `ppo_trainer.yaml` 的
> `trainer.diversity_metrics` 块。本目录负责全 L 扫描与跨模型离线对比。

## 方法

对每个 (mode, model, bench):

1. 按 `example_id` 分组 (每题 R=16 条采样，`t0.7_p0.95`)。
2. 每条 response 用 **Qwen3 tokenizer**(`vocab=151669`) 切 token，保留前
   `L` 个 token，取其中**去重后的 token 级 4-gram 集合** `G4`。
3. 两两 Jaccard 距离：`d_J(i,j) = 1 − |G4_i ∩ G4_j| / |G4_i ∪ G4_j|`。
4. 题内对 i<j 全对取平均 (即论文的 `2/(R(R−1)) Σ` 加权),再对题目宏平均得到 `D_J(L)`。**值越大越多样**。

与论文的两处差异：论文每题 R=32，这里数据是 n16(R=16);论文只算 AIME24，这里覆盖 aime24 / aime25 / amc23(amc23 文件含 83 题)。

### L 网格 (按实际 max length 重新设计)

论文用固定 `{512, 1024, 2048, 4096, Full}`。本数据的生成上限是
**31744 tokens**(文件名后缀 `MNT31744` = max new tokens),实测长度分布
(见 [`results/length_stats.csv`](results/length_stats.csv)):

- thinking 模式 AIME 上 **p50 = 31744**,即过半回答顶格截断;step checkpoint
  顶格率 **70-90%**;no-thinking 顶格率 1-62%(Base 模型最高)。
- 因此论文原网格中 `Full` 主要由"被截断的回答"主导，且各模型顶格率差异很大，
  直接比较会被截断行为污染。

本仓库采用 `{256, 512, 1024, 2048, 4096, 8192, 16384, Full}`:

- 新增 256(覆盖 amc23 no-thinking 的短回答，p50≈1.3k) 与 8192、16384——
  16384 是上限内最大的"干净"测量点 (所有模型绝大多数回答都超过它),
  其余按 2 倍步进覆盖到接近上限。
- 读数建议：跨模型比较优先看 **L ≤ 16384 的段**(受截断影响小);
  `Full` 列务必结合 `cap_hit_rate` 一起解读。
- 若要进一步消除"回答长短本身"的影响，可改为每条回答按自身长度的分位数位置截断
  (未实现，当前保持与论文可比的绝对 token 网格)。

## 两种条件

- `all`:全部 16 条 response 参与配对。
- `no-repetition`:先用 [`repetition/`](../repetition/) 的 zlib 窗口检测器
  (window=1024, stride=256, threshold=10) 标记 repetition response，剔除后再配对;
  剩余不足 2 条的题跳过 (本数据中未出现)。

注意:repetition 检测是 **response 内部**的循环;Jaccard 是 **response 之间**的距离，
两者度量不同的东西。Base 模型 rep 率高但 D_J 也高——重复循环的退化文本彼此差异很大，
会**虚增** D_J;step checkpoint 尤其明显 (见下表),这正是需要 `no-repetition` 对照的原因。

## 运行

```bash
# 计算(.venv 需装 matplotlib / numpy / tokenizers;tokenizers 已装)
.venv/bin/python Jaccard/jaccard_diversity.py --workers 6

# 画图(独立模型图自动排除 global_step_*;另有 steps 对比图)
.venv/bin/python Jaccard/plot_jaccard.py

# 分阶段计时(见下方"计算开销")
.venv/bin/python Jaccard/benchmark.py
```

Tokenizer 默认使用 HF 缓存中 tiny-random-qwen3 快照内的原版 Qwen3 `tokenizer.json`
(权重随机，tokenizer 文件与 Qwen3 官方一致),可用 `--tokenizer` 覆盖。

## 计算开销

`benchmark.py` 在真实 rollout 抽样上分阶段计时 (单进程、热缓存，Apple Silicon):

| 阶段                                                              | 延迟 / 1k tokens                   |
| ----------------------------------------------------------------- | ---------------------------------- |
| json.loads                                                        | ≈0.005 s/MB(可忽略)                |
| tokenize(Rust tokenizers，单线程)                                 | 0.8-0.9 ms                         |
| repetition 扫描 (zlib 窗口)                                       | 0.15-0.23 ms                       |
| 4-gram 列表构建                                                   | 0.08-0.14 ms                       |
| per-L frozensets(8 个 L)                                          | 0.11-0.13 ms                       |
| pairwise + 均值 (16 条 × 8 个 L)                                  | ~0.75 ms                           |
| **各阶段合计**                                                    | **~2.2 ms / 1k tokens**            |
| **端到端实测**(全量 522M tokens / 48 组合，6 workers,24 min wall) | **~16.6 ms / 1k tokens(单核等效)** |

端到端比热缓存阶段之和高约一个量级，来自：全量文件的 JSON 解析与 IO、
6 个 worker 同时持有 GB 级 4-gram 集合的内存压力、进程启动，以及
step checkpoint 等长回答组合上更大的 pairwise 集合。362k tokens/s 的吞吐意味着
重跑一次全量分析约 25 分钟;若只加模型/加 bench，按 ~2.2 ms/1k tokens 的阶段成本估算即可。

另外 `grading_results.json` 里 eval 框架自带一个 corpus 级 `distinct_4gram`
(唯一 4-gram 占比),与本文的 pairwise D_J 是同一素材的两种聚合方式，可交叉验证。

## 输出

- [`results/jaccard_diversity.csv`](results/jaccard_diversity.csv):48 个
  (mode, model, bench)× 2 条件 × 8 个 L 的长表，含 `n_problems_used`、
  `n_responses`、`repetition_rate`。
- [`results/length_stats.csv`](results/length_stats.csv):每个组合的 token 长度
  p50/p75/p90/p99/max 与顶格率 `cap_hit_rate`(≥31744)。
- `figures/all/`、`figures/repetition-filtered/`:两种条件各自的
  per-(mode, bench) 单图 + 2×3 总览;同一模型跨图颜色/标记一致
  (plot-skill `tableau-paper` preset)。
- `figures/repetition-filtered/steps_vs_1p7b-base_4b.png`:7 个
  `global_step_*` checkpoint(Blues 渐变，越深步数越大) 与
  Qwen3-1.7B-Base(no-thinking)、Qwen3-4B(thinking) 在三个 bench 上的
  repetition-filtered 对比。

## 结果：独立模型 (L = Full)

| 模式        | 模型                   | bench  | cap% | rep% | D_J@16384 filt | D_J full all | D_J full filt |
| ----------- | ---------------------- | ------ | ---- | ---- | -------------- | ------------ | ------------- |
| no-thinking | Qwen3-1.7B             | aime24 | 9    | 8.3  | 0.749          | 0.752        | 0.750         |
| no-thinking | Qwen3-1.7B             | aime25 | 9    | 6.2  | 0.700          | 0.700        | 0.700         |
| no-thinking | Qwen3-1.7B             | amc23  | 2    | 1.3  | 0.652          | 0.653        | 0.652         |
| no-thinking | Qwen3-1.7B-Base        | aime24 | 59   | 46.2 | 0.868          | 0.911        | 0.867         |
| no-thinking | Qwen3-1.7B-Base        | aime25 | 62   | 54.2 | 0.825          | 0.907        | 0.825         |
| no-thinking | Qwen3-1.7B-Base        | amc23  | 49   | 46.8 | 0.841          | 0.900        | 0.841         |
| no-thinking | Qwen3-4B               | aime24 | 26   | 10.8 | 0.694          | 0.699        | 0.696         |
| no-thinking | Qwen3-4B               | aime25 | 21   | 9.4  | 0.553          | 0.566        | 0.553         |
| no-thinking | Qwen3-4B               | amc23  | 6    | 2.1  | 0.599          | 0.600        | 0.600         |
| no-thinking | Qwen3-4B-Base          | aime24 | 44   | 40.0 | 0.570          | 0.726        | 0.572         |
| no-thinking | Qwen3-4B-Base          | aime25 | 45   | 46.0 | 0.814          | 0.902        | 0.814         |
| no-thinking | Qwen3-4B-Base          | amc23  | 47   | 47.6 | 0.831          | 0.908        | 0.831         |
| no-thinking | Qwen3-4B-Instruct-2507 | aime24 | 11   | 11.5 | 0.588          | 0.595        | 0.589         |
| no-thinking | Qwen3-4B-Instruct-2507 | aime25 | 10   | 4.8  | 0.507          | 0.514        | 0.507         |
| no-thinking | Qwen3-4B-Instruct-2507 | amc23  | 1    | 1.1  | 0.490          | 0.492        | 0.490         |
| no-thinking | Qwen3-8B               | aime24 | 21   | 7.9  | 0.689          | 0.693        | 0.691         |
| no-thinking | Qwen3-8B               | aime25 | 25   | 10.8 | 0.559          | 0.561        | 0.559         |
| no-thinking | Qwen3-8B               | amc23  | 6    | 4.1  | 0.640          | 0.640        | 0.640         |
| thinking    | Qwen3-1.7B             | aime24 | 16   | 8.3  | 0.740          | 0.758        | 0.750         |
| thinking    | Qwen3-1.7B             | aime25 | 23   | 11.2 | 0.737          | 0.743        | 0.747         |
| thinking    | Qwen3-1.7B             | amc23  | 8    | 5.9  | 0.408          | 0.418        | 0.422         |
| thinking    | Qwen3-4B               | aime24 | 11   | 5.8  | 0.711          | 0.722        | 0.715         |
| thinking    | Qwen3-4B               | aime25 | 18   | 6.0  | 0.774          | 0.784        | 0.781         |
| thinking    | Qwen3-4B               | amc23  | 6    | 3.5  | 0.626          | 0.628        | 0.628         |
| thinking    | Qwen3-8B               | aime24 | 11   | 4.0  | 0.800          | 0.804        | 0.803         |
| thinking    | Qwen3-8B               | aime25 | 21   | 4.4  | 0.753          | 0.767        | 0.766         |
| thinking    | Qwen3-8B               | amc23  | 9    | 5.6  | 0.455          | 0.445        | 0.463         |

## 结果:step checkpoints(1.7B-thinking 学生 ← 4B-thinking teacher)

| 模式     | 模型     | bench  | cap% | rep% | D_J@16384 filt | D_J full all | D_J full filt |
| -------- | -------- | ------ | ---- | ---- | -------------- | ------------ | ------------- |
| thinking | step 20  | aime24 | 70   | 65.0 | 0.912          | 0.941        | 0.913         |
| thinking | step 20  | aime25 | 75   | 63.3 | 0.855          | 0.867        | 0.856         |
| thinking | step 20  | amc23  | 75   | 73.8 | 0.711          | 0.781        | 0.711         |
| thinking | step 40  | aime24 | 79   | 62.5 | 0.413          | 0.711        | 0.413         |
| thinking | step 40  | aime25 | 77   | 64.4 | 0.740          | 0.868        | 0.741         |
| thinking | step 40  | amc23  | 76   | 73.3 | 0.788          | 0.848        | 0.788         |
| thinking | step 60  | aime24 | 78   | 76.0 | 0.366          | 0.768        | 0.367         |
| thinking | step 60  | aime25 | 82   | 80.2 | 0.471          | 0.879        | 0.471         |
| thinking | step 60  | amc23  | 83   | 83.4 | 0.662          | 0.771        | 0.661         |
| thinking | step 80  | aime24 | 90   | 80.6 | 0.467          | 0.839        | 0.468         |
| thinking | step 80  | aime25 | 88   | 81.5 | 0.465          | 0.850        | 0.466         |
| thinking | step 80  | amc23  | 83   | 83.4 | 0.503          | 0.775        | 0.503         |
| thinking | step 100 | aime24 | 86   | 84.4 | 0.429          | 0.839        | 0.431         |
| thinking | step 100 | aime25 | 87   | 81.9 | 0.693          | 0.922        | 0.693         |
| thinking | step 100 | amc23  | 84   | 82.7 | 0.583          | 0.805        | 0.587         |
| thinking | step 120 | aime24 | 90   | 79.6 | 0.419          | 0.813        | 0.421         |
| thinking | step 120 | aime25 | 81   | 75.4 | 0.673          | 0.898        | 0.674         |
| thinking | step 120 | amc23  | 88   | 82.7 | 0.450          | 0.831        | 0.451         |
| thinking | step 140 | aime24 | 85   | 69.4 | 0.598          | 0.806        | 0.600         |
| thinking | step 140 | aime25 | 85   | 71.9 | 0.824          | 0.901        | 0.826         |
| thinking | step 140 | amc23  | 82   | 72.7 | 0.775          | 0.877        | 0.776         |

## 结果要点

- **D_J 随 L 单调上升后趋平**:开头 token 大量共享题干复述，后文逐渐发散;
  thinking 模式的上升一直持续到 8192-16384 才饱和——旧网格 (止步 4096) 会漏掉
  一半以上的动态范围。
- **`Full` 被截断主导**:thinking AIME 的 p50 长度就是 31744 上限;step checkpoint
  顶格率 70-90%、1.7B-Base 49-62%,跨模型比较 Full 列时必须结合顶格率。
- **过滤 repetition 在高 rep 模型上剧烈改写结论**:Qwen3-4B-Base(aime24)
  0.726 → 0.572;step checkpoint 全线从虚高的 0.77-0.94 掉到 0.37-0.91
  (重复循环文本彼此差异大，虚增表观多样性);低 rep 模型基本不动。
- **step checkpoint 的真实多样性非单调**:step 20 仍接近 Base 式发散
  (0.91),step 40-120 在 aime24 上塌缩到 0.37-0.47,step 140 部分回升
  (0.60 / 0.82 / 0.78)。与 teacher Qwen3-4B(thinking，过滤后 0.71) 相比，
  训练后期的学生多样性仍低。
- **Instruct-2507 是多样性最低的独立模型**(0.49-0.60),Base 模型最高 (≥0.83)。
