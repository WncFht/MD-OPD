# Alignment analysis pipeline

`alignment/` 是 One-Shot-OPD 的离线分析代码。设计目标不是管理每一张图片，而是让**数据如何进入系统**、**指标如何计算**成为稳定的代码接口；绘图脚本只是这些指标的一个消费者。

## 设计理念

```text
外部数据 / 本地文件
        ↓
数据层：读取、排序、保留原始字段和 metadata
        ↓
指标层：distance、absorption、tail fit、spectrum、validation
        ↓
绘图和报告：展示指标表
```

数据层不解释科学含义。例如 W&B 的 `opd/token_reward_mean` 只是原始字段；它被解释成正的 distance proxy 是指标层的职责。这样更换数据来源时不必重写绘图，更换指标口径时也不必修改 loader。

指标层输出表格或 JSON，而不是直接输出图片。当前 alignment 的标准逐步表为：

| 列         | 含义                                                        |
| ---------- | ----------------------------------------------------------- |
| `step`     | 训练步                                                      |
| `d_nats`   | 正的 reverse-KL distance proxy，即 `-opd/token_reward_mean` |
| `d_rel`    | 相对于分析 baseline（默认 step 30）的距离                   |
| `v_direct` | 原始单步吸收率                                              |
| `v_geo`    | 几何窗口平滑吸收率                                          |

指标计算保留完整 step 范围。是否从 step 0 展示、哪个 baseline 用于归一化、baseline 前如何淡化，属于绘图层的决定。

## 目录结构

```text
alignment/
├── core/
│   ├── data.py       # RunHistory 和本地 history loader
│   └── metrics.py    # distance / absorption / summary
├── config/
│   └── runs.py       # M4 run 的 label、颜色、effective-time scale
├── pull_wandb_metrics.py
├── compute_alignment.py
├── plot_alignment.py
├── fit_tail.py
├── extract_paper_figures.py
├── analyze_paper_curves.py
├── plot_paper_analysis.py
├── invert_spectrum.py
├── plot_metric_theory.py
├── plot_qwen3_diagnosis.py
├── plot_figure19_m4.py
├── plot_baseline_variants.py
├── plot_m4_six_panel.py
├── plot_m4_supplement.py
├── plot_multirun_eval_fit.py
├── runs/             # W&B 原始 history 和 metadata
├── derived/          # 指标表、拟合参数、谱反演结果
├── paper_curves/     # 从论文 PDF 提取的 tidy 曲线
└── figures/          # 绘图输出
```

`core/` 是可复用的数据和指标代码，`config/` 是实验组配置；根目录脚本是面向研究流程的可执行入口。新代码应从 `core` 和 `config` 导入，不在脚本之间复制指标实现。

## 常用流程

### 1. 拉取 W&B 数据

```bash
.venv/bin/python alignment/pull_wandb_metrics.py \
  --entity wncfht --project One-Shot-OPD \
  --run-id mdxt04lm be00rj9e --out alignment/runs
```

这个步骤只做外部数据落盘，不计算 alignment 指标。已有文件默认不会重复拉取；需要更新时使用 `--force`。

### 2. 计算标准 alignment 指标

```bash
.venv/bin/python alignment/compute_alignment.py \
  --runs-dir alignment/runs --out-dir alignment/derived --t-start 30
```

输出 `derived/<run_id>__alignment.csv` 和 `derived/summary.json`。入口脚本负责遍历输入和写文件，真正的计算在 `core/metrics.py`。

### 3. 画基础 alignment 图

```bash
.venv/bin/python alignment/plot_alignment.py \
  --derived-dir alignment/derived --figures-dir alignment/figures
```

基础图会保留 step 0 到末尾。step 30 以前使用浅色，step 30 用竖线标注为分析 baseline / clipping boundary。

### 4. 尾部拟合

```bash
.venv/bin/python alignment/fit_tail.py --derived-dir alignment/derived --plot
```

会比较幂律、拉伸指数和 floor+指数，并写入 `derived/tail_fits.json`。

### 5. M4、论文和 Qwen3 分析

M4 图共享 `config/runs.py` 的 run 配置，并通过 `core.data` 和 `core.metrics` 取得数据和 distance：

```bash
.venv/bin/python alignment/plot_figure19_m4.py
.venv/bin/python alignment/plot_m4_supplement.py
.venv/bin/python alignment/plot_m4_six_panel.py
```

论文曲线和 Qwen3 诊断有独立数据源，不为了形式统一而强行塞进同一种 history schema：

```text
论文 PDF → extract_paper_figures.py → paper_curves/*.csv
Qwen3 eval JSONL + W&B history → plot_qwen3_diagnosis.py
```

它们可以共享 tail fitting、absorption 和 plotting style，但保留各自 loader 更清楚。

## 修改代码时的约定

1. 新数据源先加 loader，不要在绘图脚本里直接读取原始文件。
2. 新指标先加到 `core/metrics.py`，返回数组、DataFrame 或 JSON-compatible 结果。
3. 绘图函数接收已经计算好的数据，绘图层负责 baseline、坐标轴、颜色和输出路径。
4. 不要在多个脚本重复定义 run id、label、color 和 scale；M4 配置放在 `config/runs.py`。
5. 保留现有脚本作为可执行入口，但新功能应避免 import 时自动画图。
6. 指标口径发生变化时，先在 docstring 和 README 说明，不要默默改变已有 derived 文件的含义。

## 后续 TODO

### 优先级高

- 把剩余 M4 图中的 `step >= 30` / `step >= 60` 处理统一到绘图辅助函数，全面遵守 step 0 + baseline 淡化约定。
- 把 `fit_tail.py` 的拟合函数迁入 `core/tail.py`，让论文曲线和本地 run 共用同一套 tail API。
- 把 `plot_qwen3_diagnosis.py` 中的绝对路径改成命令行参数或项目配置。
- 给 `core.metrics` 增加少量端到端检查：输入已有 CSV，确认标准列、baseline 和 summary 数值稳定。

### 优先级中

- 将 `plot_metric_theory.py` 拆成理论合成图和真实数据诊断图两个入口；先移动职责，不改变数学和图形语义。
- 为不同 distance 口径增加显式 metric 选择，而不是在脚本中隐含 `-opd/token_reward_mean`。
- 把 validation、CE/entropy/KL 分解、effective-time folding 提炼成指标函数，绘图只消费结果。
- 统一脚本的 `main()` 和参数风格，消除 import 时直接执行绘图的入口。

### 优先级低

- 在数据和指标接口稳定后，再考虑 `python -m alignment ...` 的统一 CLI。
- 如果未来需要跨项目复用，再把 W&B 拉取和核心 metric 模块整理成可安装 package；当前不必过早包装。

## 当前边界

`alignment/` 是离线分析工具，不负责 GPU 训练、远程集群提交或 checkpoint 修改。`spectral_effopd.py` 是 trainer-agnostic 的算法原型，和离线绘图 pipeline 共享研究背景，但不应被误认为是数据分析入口。
