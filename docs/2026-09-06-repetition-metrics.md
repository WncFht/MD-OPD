# 2026-09-06: 训练/验证过程中的 repetition 与 compression 指标

## 背景

rollout repetition 的离线分析（`repetition/analyze_and_plot.py`，滑窗 zlib 检测）此前只作用于 eval 输出。现在把同一套检测器以监控指标的形式接入 verl 训练循环，train 和 valid 每次都能看到退化信号，不再依赖事后分析。检测器实现独立放在 vendored verl 内，`repetition/` 目录的脚本不参与训练。

## 检测方法

`verl/verl/trainer/ppo/repetition_metrics.py`：对每条解码后的 response 做重叠字符窗口扫描，窗口 1024 字符、步长 256，每个窗口计算 zlib level 9 压缩比（原始字节 / 压缩后字节），任一窗口 >= 10 判为 repetition。窗口逻辑与离线分析逐窗一致（有 34 条合成文本的逐窗对照测试）。

## 指标

train 侧（`repetition/*`、`compression/*` 前缀，按 step 记录，分母排除 aborted 样本）：

- `repetition/frac`：被标记样本占比
- `repetition/max_window_ratio/{mean,p90,max}`：最大局部压缩比的聚合，连续量，比二值标记更早暴露退化
- `repetition/onset_frac/mean`：首次超标窗口起点 / response 长度，未标记记 1.0
- `repetition/post_onset_frac/mean`：onset 之后的字符占比，未标记记 0.0，即平均浪费的生成比例
- `compression/ratio/mean`：全文 zlib 压缩比均值

valid 侧（组级通道，`val-aux/{data_source}/...`，按 uid 分组、跨 prompt 平均；分母排除 aborted/空文本，与 train 对齐）：

- `repetition/frac`：每题被标记比例的宏平均
- `repetition/all_flagged`：整题全崩率（题内所有样本都被标记）；这是原来 worst@N bootstrap 想近似但算不对的量
- `repetition/max_window_ratio/{mean,p90,max}`：p90/max 是题内 n 条样本的分位数/极值再跨题平均
- `repetition/{onset_frac,post_onset_frac}/mean`
- `compression/ratio/mean`

命名与 train 侧一致。valid 有意不走 `process_validation_metrics` 的 per-sample 通路：repetition 是 1=坏的二值旗标或连续量，per-sample 通路的 pass@k（"至少一条坏"或恒 0）与 best@N/worst@N bootstrap（mean 的单调变换或写反的 max/min 曲线）全是噪声——每个 bench 会多出约 100 条无意义序列和百万次 bootstrap 调用。同类的组级指标还有 jaccard/answer（`trainer.diversity_metrics`，见 `diversity_metrics.py`）。

## 配置

`ppo_trainer.yaml` 的 `trainer.repetition_metrics` 块（megatron yaml 继承同款）：

```yaml
repetition_metrics:
    enabled: True # 总开关
    interval: 1 # train 侧每 K 个 step 算一次；valid 不受 interval 影响
    window_size: 1024
    stride: 256
    threshold: 10.0
```

## 接入点

- train：`ray_trainer.py` fit 中 `compute_data_metrics` 之后调用 `RayPPOTrainer._compute_repetition_metrics`，解码非 aborted 的 `responses` 后聚合。
- valid：`_validate` 中解码 `output_texts` 后逐条算标量，经 `compute_grouped_repetition_metrics` 按 uid 分组聚合成每题一个值，写入 `group_infos_dict`（与 jaccard/answer 同一条组级通道），在 `_val_metrics_update` 中跨 prompt 平均输出；aborted/空文本不计入分母。

## 成本

每条 response 约 80 个窗口、每窗口约 6.5 us（本机 zlib benchmark），压缩扫描本身约 0.5 ms/response；train 侧开销大头是解码，每步 1-2 s 量级，相对整步训练可忽略。
