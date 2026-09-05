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

valid 侧（`val-aux/{data_source}/.../mean@N`，随现有按 data_source 分组聚合；开 `validation_data_dir` 时也会写入 dump JSONL）：

- `repetition`、`repetition_max_window_ratio`、`repetition_onset_frac`、`repetition_post_onset_frac`、`response_compression_ratio`

valid 的 onset/post_onset 用 1.0/0.0 哨兵值定义在全样本上，因此直接走 `process_validation_metrics` 的 mean@N 通路，无需单独的 flagged 子集统计。

## 配置

`ppo_trainer.yaml` 的 `trainer.repetition_metrics` 块（megatron yaml 继承同款）：

```yaml
repetition_metrics:
  enabled: True     # 总开关
  interval: 1       # train 侧每 K 个 step 算一次；valid 不受 interval 影响
  window_size: 1024
  stride: 256
  threshold: 10.0
```

## 接入点

- train：`ray_trainer.py` fit 中 `compute_data_metrics` 之后调用 `RayPPOTrainer._compute_repetition_metrics`，解码非 aborted 的 `responses` 后聚合。
- valid：`_validate` 中解码 `output_texts` 后逐条算标量，塞进 `reward_extra_infos_dict`，复用按 uid/data_source 的分组聚合；每个样本都必须有值（空文本按"无 repetition"计）。

## 成本

每条 response 约 80 个窗口、每窗口约 6.5 us（本机 zlib benchmark），压缩扫描本身约 0.5 ms/response；train 侧开销大头是解码，每步 1-2 s 量级，相对整步训练可忽略。
