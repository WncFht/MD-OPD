# 反思词（lexical reflection）统计

本目录提供一个可复现的 OPD 评测 rollout 反思词统计脚本。它会扫描 `no-thinking` 和
`thinking` 两种模式，先沿用仓库现有的 zlib 局部压缩率复读检测器剔除被判定为复读的
rollout，再对剩下的 rollout 统计每个反思词的出现次数，并给出按千词归一化的 rate。

## 运行

```bash
python3 reflection/lexical_count.py \
  --eval-root /path/to/justrl_eval_outputs \
  --output-dir reflection/results \
  --top-k 30
```

- `--exclude-checkpoints`：跳过名字形如 `global_step_100` 的训练中间 checkpoint 目录。
  默认保留，并把它们当作独立模型标签输出。
- `--repetition-window` / `--repetition-stride` / `--repetition-threshold`：复读检测参数，
  默认与仓库检测器一致（1024 / 256 / 10.0）。

## 输出

- `per_rollout.csv`：每个未被判为复读的 rollout 一行，包含总反思词数、分类计数，
  以及按千词归一化的 rate。
- `model_summary.csv`：每个 `(mode, model)` 的总 rollout 数、复读数、保留数、复读率，
  以及模型级的均值和分类计数。
- `word_stats.csv`：逐词细粒度统计（语料级）。词表中的每个词在每个 `(mode, model)`
  下都有一行（计数为 0 的词也保留），字段为 `mode, model, word, category, count,
rate_per_1k_words`。
- `word_response_stats.csv`：逐词 × 逐 response 细粒度统计。同样覆盖词表中每个词
  （含 0 计数），字段为 `mode, model, word, category, count, responses_with_word,
share_of_responses`（包含该词的 response 占比）、`mean_count_per_response`（每条
  response 平均出现次数）、`mean_count_when_present`（出现时的平均次数）和
  `mean_rate_per_1k_words`（per-response 口径的平均每千词 rate）。
- `top_keywords.csv`：每个 `(mode, model)` 出现次数最高的 top-k 词（默认 30）。
- `lexical_list.json`：本次运行实际使用的完整词表。
- `metadata.json`：复读检测与匹配配置。

## 词表说明

词表刻意取得比较宽，分成 `core`（核心反思）、`branch`（分支/转折）、
`verification`（验证）和 `transition`（过渡）四类。匹配大小写不敏感、按词边界对齐、
长短语优先，避免把 `checker` 里的 `check` 误算进去。

要一个高精度的主指标，用 `count_core`（可以再加 `count_verification`）；
`count_transition` 是更宽的辅助统计，因为 `so`、`now`、`first` 这类词在普通叙述中
也会大量出现。

## 画图

`plot_word_rates.py` 复用 `.agents/skills/plot-skill` 的样式 preset，把
`word_stats.csv` 画成分组柱状图：横轴是词（按类别分成四个子图），每个模型一个系列，
纵轴是每千词 rate。

```bash
python3 reflection/plot_word_rates.py \
  --stats reflection/results/word_stats.csv \
  --out-dir reflection/figs
```

- 输出 `word_rates_no-thinking.png` 和 `word_rates_thinking.png` 两张图。
- 名字含 `global_step_N` 的训练中间 checkpoint 默认不画（这类模型大部分 rollout 已被
  复读检测剔除），需要时加 `--include-checkpoints`。
- 只画在该模式下至少出现过一次的词；计数全为 0 的词不出现在图中，
  完整数据见 `word_stats.csv`。

### Per-response 统计图

`plot_per_response.py` 读取 `per_rollout.csv`（其中本来就只包含通过复读检测的
rollout，repeat 已剔除），对每个模型画四项 response 级指标的柱状图：
平均反思词数、中位反思词数、平均每千词 rate，以及含至少一个反思词的
response 占比。

```bash
python3 reflection/plot_per_response.py \
  --per-rollout reflection/results/per_rollout.csv \
  --out-dir reflection/figs
```

- 输出 `per_response_no-thinking.png` 和 `per_response_thinking.png` 两张图。
- 模型配色与 `word_rates_*.png` 保持一致，同名模型颜色相同。
- 训练中间 checkpoint 同样默认不画，可用 `--include-checkpoints` 打开。

### 逐词 per-response 统计图

`plot_word_response.py` 读取 `word_response_stats.csv`，把每个词的 response 级指标
画成分组柱状图（布局与 `word_rates_*.png` 一致：按类别分面、每个模型一个系列），
每个模式两张图：

- `word_presence_{mode}.png`：纵轴为**包含该词的 response 占比（%）**，即每条
  response 有多大概率出现这个词；
- `word_count_per_response_{mode}.png`：纵轴为**每条 response 中该词的平均出现次数**。

```bash
python3 reflection/plot_word_response.py \
  --stats reflection/results/word_response_stats.csv \
  --out-dir reflection/figs
```

checkpoint 默认排除，可用 `--include-checkpoints` 打开；只画至少出现过一次的词。
