# 2026-09-06: 训练/验证过程中的 lexical reflection（反思词）指标

## 背景

`reflection/lexical_count.py` 此前只对 eval rollout 做离线反思词统计。离线结果
（`reflection/results/`）显示：训练退化时反思词是被**选择性抹除**的——崩溃的
`global_step_100` checkpoint 里 `wait`、`hmm`、`let me check` 的 rate 恰好归零，
而 `alternatively` 从 ~2.6/1k 掉到 ~0；这通常早于 zlib 复读旗标触发。现在把同一
套词表匹配逻辑以监控指标的形式接入 verl 训练循环，train 和 valid 每次都能看到
反思词的退化信号，不再依赖事后分析。

## 词表

`verl/verl/trainer/ppo/reflection_metrics.py` 的 `LEXICAL_LIST`：**40 个标志词，
三个类目**（在线 mini-panel）。

- `core`（13）：wait, hmm, hold on, oh, ah, actually, oops, on second thought,
  another thought, think again, reconsider, let me reconsider, rethinking
- `branch`（14）：but, however, alternatively, alternative, another approach,
  another way, instead, rather, on the other hand, or perhaps, maybe, perhaps,
  but then, but wait
- `verification`（13）：verify, check, double-check, double check, recheck,
  recalculate, confirm, validate, make sure, let me check, ensure, cross-check,
  cross check

词表是离线清单（`reflection/lexical_count.py`，含 transition 类）的精选子集：
去掉语料级冗余变体（`let's check`/`lets check`、`i should rethink` 等）和纯填充
过渡词（`so`/`now`/`first`，正常叙述里大量出现，作为在线指标是噪声）；`but` 是
刻意加入的——普通叙述也常用，但它的塌缩属于观测到的退化特征。离线脚本保留自己
更宽的词表不变，保证已生成的离线结果可复现。

匹配与离线逐条一致：大小写不敏感、词边界对齐（`checker` 不算 `check`、`await`
不算 `wait`）、长短语优先（`but wait` 记作一个短语，不再重复计入 `wait`）。

## 指标

每条 response 先归一化再聚合：rate 按千个空白分词归一，presence 是 0/1 出现
旗标，因此 batch/题组聚合都是宏平均，response 长度漂移不会污染数字。

与离线统计的口径差异：离线 `lexical_count.py` 会先剔除 repetition 命中的
rollout 再统计，线上的 raw 序列**不剔除**（退化样本本身没有反思词，把它们计入
只会让 raw 序列的下降更陡，符合 early-warning 用途）。需要与离线数字严格对表
时，用 valid 侧的 `*_unflagged` 序列（见下）。

train 侧（`reflection/*` 前缀，按 step 记录，分母排除 aborted 样本）：

- `reflection/{core,branch,verification}/rate_per_1k`：类目级每千词出现次数
- `reflection/{core,branch,verification}/presence`：含至少一个该类词的 response 占比
- `reflection/word/<word>/rate_per_1k`、`reflection/word/<word>/presence`：
  mini-panel 逐词双口径（如 `reflection/word/wait/rate_per_1k`、
  `reflection/word/double_check/presence`；空格折叠为下划线，连字符保留，
  `double-check` 与 `double check` 是两条独立序列）

valid 侧（组级通道，`val-aux/{data_source}/...`，按 uid 分组、跨 prompt 平均，
分母排除 aborted/空文本，与 train 对齐）：

- 与 train 同名的全部序列（题内 n 条样本取均值再跨题平均）
- `reflection/{category}/all_absent`：整题全部样本都不含该类词的比例——反思词
  版的 `repetition/all_flagged`
- `reflection/{category}/rate_per_1k_unflagged`、
  `reflection/word/<word>/rate_per_1k_unflagged`：剔除题内 repetition 命中样本
  后的 rate，与离线"先剔除 flagged rollout 再统计"的口径直接可比。题内全部
  非 aborted 样本都被 flag 时该题不贡献 unflagged 值（这些序列跨题平均的分母
  是能贡献的题）；`reflection/unflagged_groups` 记录能贡献的题占比，是
  unflagged 序列分母的健康度指标
- unflagged 序列依赖 repetition flags：repetition_metrics 开启时用其配置的
  检测器参数，关闭时退回默认参数（1024/256/10.0，即离线分析的常量）补算，
  unflagged 序列始终可用

与 repetition 相同，valid 有意不走 `process_validation_metrics` 的 per-sample
通路：反思词 rate/presence 不是 {0,1} 成功值，pass@k / best@N bootstrap 对它们
没有意义。

## 配置

`ppo_trainer.yaml` 的 `trainer.reflection_metrics` 块（megatron yaml 继承同款）：

```yaml
reflection_metrics:
    enabled: True # 总开关
    interval: 1 # train 侧每 K 个 step 算一次；valid 不受 interval 影响
```

## 接入点

- train：`ray_trainer.py` fit 中调用 `_compute_train_text_metrics`（原
  `_compute_repetition_metrics` 扩展而来）：解码一次非 aborted 的 `responses`，
  repetition 与 reflection 两个家族共享这次解码，各自尊重自己的
  enabled/interval，未启用或被 interval 跳过时不产生解码开销。
- valid：`_validate` 中对已解码的 `output_texts` 调
  `analyze_reflection_responses` + `compute_grouped_reflection_metrics`，经
  `group_infos_dict`（与 repetition、jaccard/answer 同一条组级通道）在
  `_val_metrics_update` 中跨 prompt 平均输出。repetition flags 直接复用
  repetition 家族已算好的 per-response 结果（关闭时才用默认参数补算一次），
  unflagged 序列没有额外解码或压缩开销。

## 成本

词表匹配是单趟正则：6KB 文本约 0.5 ms/response；train 侧开销大头仍是解码，
且与 repetition 共享一次 `batch_decode`，整步训练中可忽略。

## 测试

`verl/tests/trainer/test_reflection_metrics.py`：词表不变量（≥32 词、必需词覆盖、
metric key 无碰撞）、边界/长短语优先匹配、rate/presence 数值、train 宏平均、
valid 组级聚合与 valid 掩码、unflagged 序列（剔除 flagged 样本、全 flagged 题
跳过、无 flags 时不产出、flags 对齐校验）、uid/data_source 必填校验、空输入。
