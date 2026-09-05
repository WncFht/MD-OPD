# Rollout Repetition 分析

这里集中保存 rollout repetition 的检测脚本、统计结果和图片。当前不再维护多个 `v2`、`v3` 或临时分析目录。

## 检测方法

每条 response 使用重叠字符窗口扫描：

```text
window_size = 1024
stride = 256
threshold = 10
```

每个窗口计算 `compression_ratio = 原始 UTF-8 字节数 / zlib 压缩后字节数`。只要任意窗口达到阈值，就把整条 response 标记为 repetition。该方法检测单条 response 内部的低信息量循环，不是同一道题不同 response 之间的文本相似度。

## 运行

```bash
python repetition/analyze_and_plot.py \
  --eval-root /Users/fanghaotian/Desktop/src/OPD/justrl_eval_outputs \
  --output-dir repetition \
  --sample-per-model 500
```

脚本扫描 `no-thinking` 和 `thinking` 下的独立模型目录，自动排除名称包含 `global_step_*` 的训练 checkpoint。每个模型最多抽取 500 条轨迹用于曲线图；repetition rate 使用该模型目录中的全部 rollout 统计。

## 输出文件

- [`summary.csv`](summary.csv)：每个模型的 rollout 数、repetition 数、repetition rate，以及 first repetition position 的统计。
- [`sample_manifest.csv`](sample_manifest.csv)：曲线图使用的抽样轨迹及其检测结果。
- [`figure_index.json`](figure_index.json)：图文件索引。
- [`analyze_and_plot.py`](analyze_and_plot.py)：统一分析入口。

## 图片

- [`figures/repetition_rate.png`](figures/repetition_rate.png)：不同独立模型的 repetition rate 柱状图。
- [`figures/compression_ratio_distribution.png`](figures/compression_ratio_distribution.png)：最大局部 compression ratio 的区间分布柱状图。
- [`figures/max_compression_ratio_ecdf.png`](figures/max_compression_ratio_ecdf.png)：最大局部 compression ratio 的累计分布。
- [`figures/first_position_distribution.png`](figures/first_position_distribution.png)：first repetition position 和归一化位置的分布。
- [`figures/five_repetition_examples.png`](figures/five_repetition_examples.png)：5 条 repetition 与 5 条 non-repetition rollout 的窗口曲线示例。

每个模型还各有两张 response-length 曲线图：

- `*_compression_relative.png`：横轴为窗口起点除以 response 长度，适合比较不同长度的 response。
- `*_compression_absolute.png`：横轴为窗口绝对起点（千字符），适合观察 repetition 在真实字符位置的出现情况。

每张模型图包含最多 500 条淡色轨迹、median、p90 和 `threshold=10` 参考线。模型对应的全部图片可在 [`figures/`](figures/) 中查看。

## 历史图片功能

之前的图片功能没有被有意删除，已经合并回统一脚本和统一 `figures/` 目录：compression ratio 分布、ECDF、first-position 分布、5 条正例/5 条反例窗口曲线，以及每个模型的绝对/相对位置曲线都会由同一个入口重新生成。
