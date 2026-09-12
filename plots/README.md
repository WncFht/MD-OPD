# plots/

One-Shot-OPD 的离线分析与画图包。设计目标不是管理每一张图片，而是让**数据如何进入系统**、**指标如何计算**、**图怎么画**成为三层稳定接口；每个 pipeline 只是这些接口的一个消费者。

```text
W&B / 本地文件
      ↓  core/data.py       读 history CSV + meta，不解释含义
      ↓  core/metrics.py    distance / absorption / validation / fold
      ↓  core/tail.py       tail-form 拟合、local alpha
      ↓  core/accuracy.py   student/teacher baseline、自适应纵轴
      ↓  pipelines/<topic>/ 编排 panel，不重复实现指标
      ↓  figures/<topic>/ + derived/<topic>_summary.json
```

数据层不解释科学含义：W&B 的 `opd/token_reward_mean` 只是原始字段，把它解释成正的 distance proxy 是指标层的职责。换数据源不用重写绘图，换指标口径也不用改 loader。

## 目录结构

```text
plots/
├── core/
│   ├── data.py       # RunHistory、history loader、VALIDATION_TASKS
│   ├── metrics.py    # 指标表：d / d_rel / v_direct / v_geo / alpha_local 等
│   ├── tail.py       # 幂律/拉伸指数/floor+exp 拟合、local_alpha
│   ├── accuracy.py   # accuracy 纵轴自适应、student/teacher baseline
│   └── plotting.py   # figure_path、plot_series、mark_baseline、style_axis
├── config/
│   └── runs.py       # 实验组配置：M4 run 的 label、颜色、effective-time scale
├── pipelines/        # 每个 topic 一个目录
│   ├── alignment/main.py          # 基础 d/v 图（读 derived/ 指标表）
│   ├── theory/main.py             # 理论合成图 + 真实数据诊断
│   ├── qwen3/main.py              # Qwen3 eval + W&B 联合诊断
│   ├── offpolicy_exp/main.py      # off-policy 对比实验
│   ├── md_opd/main.py             # PPO epochs / MDPO 对比
│   ├── lr_ablation/*.py           # M4 lr 消融（原 m4 模块）
│   └── paper/plots.py             # 论文曲线对照
├── pull_wandb_metrics.py   # W&B history → runs/*.csv
├── compute_alignment.py    # runs/*.csv → derived/*__alignment.csv
├── fit_tail.py             # derived/ → tail_fits.json + figures/tail/
├── extract_paper_figures.py / analyze_paper_curves.py
├── invert_spectrum.py      # 谱反演 → derived/ + figures/paper/
├── runs/             # W&B 原始 history 和 metadata
├── derived/          # 指标表、拟合参数、谱反演结果
├── paper_curves/     # 从论文 PDF 提取的 tidy 曲线
└── figures/          # 绘图输出，按 topic 建子目录
```

## Topic 约定

- 一个 topic = `pipelines/<topic>/` 目录 + `figures/<topic>/` 输出目录 + `derived/<topic>_summary.json` 摘要，三处共用同一标识。例如 `pipelines/offpolicy_exp/` → `figures/offpolicy-exp/` + `derived/offpolicy_exp_summary.json`。
- 新增对比实验：先定 topic 名 → `pipelines/<topic>/` 下放代码 → `figure_path("<topic>", ...)` 生成文件名。文件名约定为 `<metric>_vs_<x_axis>__<scope>[__<variant>].png`，scope 标记数据范围（如 `runs-m4`），variant 标记同族变体（如 `norm-d30`）。
- pipeline 必须是可重复执行的入口：`python -m plots.pipelines.<topic>.<module>` 从 `runs/` 重算出全部输出；不要在 import 时画图。

## 共享件速查

topic 代码只编排 panel；下面这些操作一律用 `core/` 的现成函数，不要自己再写一遍：

| 需求                        | 入口                                                                       |
| --------------------------- | -------------------------------------------------------------------------- |
| 加载 + 算指标一次完成       | `load_run_metrics(run_id, runs_dir)` → `(history, metrics)`                |
| 选 d_t 口径                 | `compute_alignment_metrics(..., column=)` / `distance_proxy(..., column=)` |
| 取某 step 的值 / 归一化     | `value_at_step` / `normalize_at_step` / `metrics.d_rel`                    |
| fold 时间轴                 | `effective_step(steps, scale)` / `folded_baseline(ax, step, scale, color)` |
| 画 step 序列（含淡化）      | `plot_series`                                                              |
| baseline 阴影 + 竖线        | `mark_baseline(ax, step)`                                                  |
| 轴收尾（y 网格/去边框/x=0） | `style_axis`                                                               |
| validation 均值 / 分任务    | `validation_mean` / `ce_entropy_kl`                                        |
| accuracy 纵轴 / 参考线      | `accuracy_ylim` / `add_accuracy_baselines`                                 |
| 输出文件名                  | `figure_path`                                                              |
| 图内 d_t 定义标注           | `distance_definition`                                                      |
| plot-skill 风格             | `load_plot_style()`                                                        |

## 指标口径

训练器在 student 采样到的 response token 上定义带符号的 OPD advantage：

$$
A_t = \log \pi_T(y_t\mid s_t)-\log \pi_\theta(y_t\mid s_t).
$$

当前目标 run 使用 `algorithm.adv_estimator=token_reward_direct`，所以这个 dense token reward 直接作为 advantage 使用，没有再减 value baseline。训练器同时记录三个不同概念：

| 日志字段                | 定义                                                     | 作用                    |
| ----------------------- | -------------------------------------------------------- | ----------------------- |
| `opd/adv_mean`          | `mean(A_t)`，带符号                                      | 正常 advantage 的均值   |
| `opd/adv_abs_mean`      | `mean(abs(A_t))`                                         | L1 型 log-prob 缺口诊断 |
| `opd/token_reward_mean` | `mean(r_t)`；当前 sampled-token 配置中与 `adv_mean` 对齐 | OPD reward 的原始均值   |

因此标准表中的距离不是 `adv_abs_mean`，而是：

$$
d_t^{\mathrm{KL\ proxy}}
  = -\operatorname{mean}(A_t)
  = -\texttt{opd/token\_reward\_mean}.
$$

对 student-sampled token，期望意义下 $\mathbb E_{\pi_\theta}[A_t] = -\mathrm{KL}(\pi_\theta\Vert\pi_T)$，所以取负后是 sampled-token reverse-KL 的 Monte Carlo proxy。它不是 full-vocabulary KL，也不能把带符号的 `adv_mean` 本身直接称为 distance：`adv_mean` 可以为负。若研究问题明确要求论文里的 L1 口径，应单独定义

$$
d_t^{\mathrm{L1}}=\operatorname{mean}(|A_t|)
$$

并在列名、图例和图内公式中显式写出；不能把 `adv_abs_mean` 静默替换当前 `d_t`。换口径时给 `compute_alignment_metrics` 传 `column=`（如 `opd/top_k_reward_mean`），不要在 pipeline 里另造指标表。

指标层输出表格或 JSON，而不是直接输出图片。标准逐步表（`compute_alignment_metrics`）的列：

| 列            | 含义                                                     |
| ------------- | -------------------------------------------------------- |
| `step`        | 训练步                                                   |
| `d_nats`      | 当前口径的 distance，即 `-<column>`                      |
| `d_rel`       | 相对于分析 baseline（默认 step 30，插值取）的距离        |
| `v_direct`    | $ (d_t-d_{t+1})/d_t $ 的原始单步吸收率                   |
| `v_geo`       | 在 $[t/1.4,1.4t]$ 上拟合 $\log d_t$ 得到的几何窗口吸收率 |
| `alpha_local` | $-\mathrm d\log d_t/\mathrm d\log t$ 的局部 log-log 指数 |

`v_geo` 和 `alpha_local` 都继承 `d_nats` 的口径。几何窗口吸收率写成

$$
v_t^{\mathrm{geo}}=1-\exp(\hat\beta_t),
\qquad
\hat\beta_t=\operatorname{slope}_{[t/1.4,\,1.4t]}(\log d).
$$

连续时间的局部指数为

$$
\alpha(t)=-\frac{\mathrm d\log d_t}{\mathrm d\log t},
$$

并可用 $\alpha(t)\approx t\,v_t$ 作几何窗口近似。由于 sampled-token 的距离估计可能在单个 step 受噪声影响而非正，`v_geo` 和 `alpha_local` 的 log 拟合会跳过非正 `d_t` 样本。

指标计算保留完整 step 范围。是否从 step 0 展示、哪个 step 作 baseline、baseline 前如何淡化，属于绘图层的决定，不由指标表截断。

## 绘图约定

### step 轴

- absolute 和 normalized 图都从 step 0 画起，不截断。
- normalized 图在 baseline step 处画竖线、给 `[0, baseline]` 加阴影，baseline 左侧的曲线段淡化（`plot_series` 的 `fade`）；absolute 图整条曲线保持正常透明度，baseline 仅作参考线。
- 折叠轴（`u=scale·t`）上每个 run 的 baseline 位置不同，用 `folded_baseline` 画各自的 dotted 竖线，不加阴影。
- 需要另一个 baseline step 时改 `baseline=` 参数（如 60），不要手写 `step >= t0` 过滤。

### 图中定义

任何包含 `d_t`、`d_t/d_{t_0}`、`v_t`、`alpha(t)` 或其派生指标的图，都必须在图内标注实际使用的 `d_t` 定义（`distance_definition` 一次写全）。L1 口径必须写成 $d_t=\operatorname{mean}(|A_t|)$ 并明确标注。图中还应区分 raw one-step `v_direct`、几何窗口 `v_geo` 和 local `alpha`，避免画成同一个指标。定义和注释放在空白区域、图例区或底部脚注，不得覆盖曲线、marker、坐标轴标签或图例；空间不足时拆行或增加底部留白。

### Accuracy 图

- 横轴从 step 0 开始；纵轴一律百分比单位，范围由 `accuracy_ylim` 给出：默认按全部有效观测点 + student/teacher baseline 自动留边；`ACCURACY_Y_LIMITS=(下限, 上限)` 可手动指定，但必须覆盖全部有效点（`accuracy_ylim` 会校验），只有展示完整概率域时才用 `(0, 100)`。
- 分任务小图默认各自按 task 缩放；设了全局 limits 才共用同一组上下限。
- 有效点和连线在 baseline 左侧**不淡化**（这是 step 轴规则的显式例外，`plot_series(..., fade=False)`）。
- 参考线统一用 `add_accuracy_baselines`：`6uorrk2k`（student）和 `u87buult`（teacher）的横向虚线；均值图用三任务均值，分任务图用对应 task 的基准值。
- 稀疏日志：有多个观测 step 时用折线连实际非空点，可带小 marker；只有一个 step 才画单点；空值不当 0、不连线。

## 常用流程

### 1. 拉取 W&B 数据

```bash
.venv/bin/python plots/pull_wandb_metrics.py \
  --entity wncfht --project One-Shot-OPD \
  --run-id mdxt04lm be00rj9e --out plots/runs
```

这个步骤只做外部数据落盘，不计算指标。已有文件默认不会重复拉取；需要更新时用 `--force`。

这些 run 是离线 `wandb sync` 上来的：任务在集群跑完后才把日志同步到 W&B，所以 `run.state`（以及 meta.json 里的 `state`）可能是 `failed`/`crashed`，即使训练本身正常完成或被抢占后已续跑。**不要**用 `state` 判断一个 run 是否失败或数据是否完整；以实际拉到的 history 步数为准。

### 2. 计算标准指标表

```bash
.venv/bin/python -m plots.compute_alignment \
  --runs-dir plots/runs --out-dir plots/derived --t-start 30
```

输出 `derived/<run_id>__alignment.csv` 和 `derived/summary.json`。入口脚本负责遍历输入和写文件，真正的计算在 `core/metrics.py`。

注意 `d_rel` 有两个来源：CSV 里的 `d_rel` 是把 `--t-start`（默认 30）烤进文件的存档快照；pipeline 里随时可以用 `normalize_at_step(steps, d, t0)` 换 baseline。改 baseline 不需要重新生成 CSV。

### 3. 画基础图

```bash
.venv/bin/python -m plots.pipelines.alignment.main \
  --derived-dir plots/derived --figures-dir plots/figures
```

### 4. 尾部拟合

```bash
.venv/bin/python -m plots.fit_tail --derived-dir plots/derived --plot
```

比较幂律、拉伸指数和 floor+ 指数，写入 `derived/tail_fits.json`，每个 run 一张拟合图到 `figures/tail/`。

### 5. 各 topic

```bash
.venv/bin/python -m plots.pipelines.lr_ablation.main        # 全部 lr-ablation 图
.venv/bin/python -m plots.pipelines.lr_ablation.baselines   # 或按模块单跑
.venv/bin/python -m plots.pipelines.offpolicy_exp.main
.venv/bin/python -m plots.pipelines.md_opd.main
.venv/bin/python -m plots.pipelines.theory.main
.venv/bin/python -m plots.pipelines.paper.plots
```

lr-ablation 的 run 配置集中在 `config/runs.py`（label、颜色、effective-time scale）。论文曲线和 Qwen3 诊断有独立数据源，不为了形式统一而强行塞进同一种 history schema：

```text
论文 PDF → extract_paper_figures.py → paper_curves/*.csv
Qwen3 eval JSONL + W&B history → pipelines/qwen3/main.py
```

它们共享 tail fitting、absorption 和 plotting style，但保留各自 loader 更清楚。

## 修改代码时的约定

1. 新数据源先加 loader，不要在绘图脚本里直接读取原始文件。
2. 新指标先加到 `core/metrics.py`，返回数组、DataFrame 或 JSON-compatible 结果；绘图函数只消费已算好的数据。
3. step 序列、baseline、fold、归一化、accuracy 纵轴走共享件（见速查表），不要在 pipeline 里重复实现。
4. 不要在多个脚本重复定义 run id、label、color 和 scale；实验组配置放在 `config/runs.py`。
5. 指标口径发生变化时，先在 docstring 和 README 说明，不要默默改变已有 derived 文件的含义。

## 边界

`plots/` 是离线分析工具，不负责 GPU 训练、远程集群提交或 checkpoint 修改。`spectral_effopd.py` 是 trainer-agnostic 的算法原型，和离线绘图 pipeline 共享研究背景，但不是数据分析入口。
