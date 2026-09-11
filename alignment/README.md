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

因此，alignment 当前标准表中的距离不是 `adv_abs_mean`，而是：

$$
d_t^{\mathrm{KL\ proxy}}
  = -\operatorname{mean}(A_t)
  = -\texttt{opd/token\_reward\_mean}.
$$

对 student-sampled token，期望意义下 $\mathbb E_{\pi_\theta}[A_t] = -\mathrm{KL}(\pi_\theta\Vert\pi_T)$，所以取负后是 sampled-token reverse-KL 的 Monte Carlo proxy。它不是 full-vocabulary KL，也不能把带符号的 `adv_mean` 本身直接称为 distance：`adv_mean` 可以为负。若研究问题明确要求论文里的 L1 口径，应单独定义

$$
d_t^{\mathrm{L1}}=\operatorname{mean}(|A_t|)
$$

并在列名、图例和图内公式中显式写出；不能把 `adv_abs_mean` 静默替换当前 `d_t`。

指标层输出表格或 JSON，而不是直接输出图片。当前 alignment 的标准逐步表为：

| 列            | 含义                                                                |
| ------------- | ------------------------------------------------------------------- |
| `step`        | 训练步                                                              |
| `d_nats`      | 当前 signed-advantage reverse-KL proxy，即 `-opd/token_reward_mean` |
| `d_rel`       | 相对于分析 baseline（默认 step 30）的距离                           |
| `v_direct`    | $ (d*t-d*{t+1})/d_t $ 的原始单步吸收率                              |
| `v_geo`       | 在 $[t/1.4,1.4t]$ 上拟合 $\log d_t$ 得到的几何窗口吸收率            |
| `alpha_local` | $-\mathrm d\log d_t/\mathrm d\log t$ 的局部 log-log 指数            |

`v_geo` 和 `alpha_local` 都继承上面的 `d_nats` 口径。几何窗口吸收率写成

$$
v_t^{\mathrm{geo}}=1-\exp(\hat\beta_t),
\qquad
\hat\beta_t=\operatorname{slope}_{[t/1.4,\,1.4t]}(\log d).
$$

连续时间的局部指数为

$$
\alpha(t)=-\frac{\mathrm d\log d_t}{\mathrm d\log t},
$$

并可用 $\alpha(t)\approx t\,v_t$ 作几何窗口近似。由于 sampled-token 的距离估计可能在单个 step 受噪声影响而非正，`v_geo` 和 `alpha_local` 的 log 拟合会跳过非正 `d_t` 样本；这不是把它们改成 `adv_abs_mean`。

指标计算保留完整 step 范围。是否从 step 0 展示、哪个 baseline 用于归一化、baseline 前如何淡化，属于绘图层的决定。

## 目录结构

```text
alignment/
├── core/
│   ├── data.py       # RunHistory 和本地 history loader
│   ├── metrics.py    # distance / absorption / validation / CE-entropy-KL / folding / summary
│   ├── tail.py       # tail-form 拟合和 local alpha
│   ├── accuracy.py   # student/teacher baseline、自适应 accuracy 纵轴
│   └── plotting.py   # figure_path、baseline 标记/淡化、plot-skill 加载
├── config/
│   └── runs.py       # M4 run 的 label、颜色、effective-time scale
├── pull_wandb_metrics.py
├── compute_alignment.py
├── fit_tail.py
├── extract_paper_figures.py
├── analyze_paper_curves.py
├── plot_paper_analysis.py
├── invert_spectrum.py
├── pipelines/        # 每个 topic 一个目录
│   ├── alignment/main.py
│   ├── theory/main.py
│   ├── qwen3/main.py
│   ├── offpolicy_exp/main.py
│   ├── md_opd/main.py
│   ├── lr_ablation/{figure19,baselines,panels,supplement,eval_fit,absorption,alpha,advantage,topk}.py
│   └── paper/plots.py
├── runs/             # W&B 原始 history 和 metadata
├── derived/          # 指标表、拟合参数、谱反演结果
├── paper_curves/     # 从论文 PDF 提取的 tidy 曲线
└── figures/          # 绘图输出，按 topic 建子目录
```

`core/` 是可复用的数据和指标代码，`config/` 是实验组配置；根目录脚本是面向研究流程的可执行入口。新代码应从 `core` 和 `config` 导入，不在脚本之间复制指标实现。`core/plotting.py` 还提供 baseline 淡化（`plot_series`）、baseline 标记（`mark_baseline`）、距离定义标注（`distance_definition`）和 `figure_path` 文件名约定，topic 代码只编排 panel。

`figures/` 下每个 topic 一个子目录，目录名、`pipelines/` 下的 topic 目录名和 derived 摘要文件名共用同一个 topic 标识，例如 `pipelines/offpolicy_exp/` 输出到 `figures/offpolicy-exp/` 和 `derived/offpolicy_exp_summary.json`，`pipelines/md_opd/` 输出到 `figures/md-opd/` 和 `derived/md_opd_summary.json`。新增对比实验时先定 topic 名，再让三处命名保持一致。

### 图中定义要求

任何包含 `d_t`、`d_t/d_{t_0}`、`v_t`、`alpha(t)` 或其派生指标的图片，都必须在图内标注实际使用的 `d_t` 定义。当前日志口径至少写成：

$$
A_t=\log\pi_T-\log\pi_\theta,
\qquad
d_t=-\operatorname{mean}(A_t)=-\texttt{opd/token\_reward\_mean}.
$$

如果图片使用 L1 口径，则必须改为 $d_t=\operatorname{mean}(|A_t|)$ 并明确标注。图中还应区分 raw one-step `v_direct`、几何窗口 `v_geo` 和 local `alpha`，避免把它们画成同一个指标。
定义和注释必须放在空白区域、图例区域或底部脚注，不得覆盖数据曲线、marker、坐标轴标签或图例；空间不足时应拆行或增加底部留白。

### Accuracy 图规则

- accuracy 图的横轴从 step 0 开始；纵轴范围由 `alignment/pipelines/offpolicy_exp/main.py` 顶部的 `ACCURACY_Y_LIMITS` 控制，单位是百分比；自适应和手动校验都由 `core/accuracy.py` 的 `accuracy_ylim` 实现，其他 pipeline 直接复用。
- valid/accuracy 保留 step 30 左侧的有效点和连线，不截断、不淡化；baseline 阴影和竖线仅作参考。
- `ACCURACY_Y_LIMITS = None` 时按所有有效观测点自动留边；设为 `(下限, 上限)` 可手动缩放。手动范围必须覆盖全部有效点，不得裁掉数据；只有需要展示完整概率域时才设为 `(0, 100)`。
- 分任务小图默认分别按各自 task 的有效观测点自动缩放；设置 `ACCURACY_Y_LIMITS` 后才使用同一组全局上下限。
- accuracy 图统一用 `6uorrk2k`（student）和 `u87buult`（teacher）的横向虚线作参考；均值图使用三项 validation 的均值，分任务图使用对应 task 的基准值。
- 有多个 step 时用折线连接实际的非空观测点，可保留小 marker；只有一个 step 时才画单点。
- W&B 稀疏日志中的空值不当作 0，也不连到空值位置。

## 常用流程

### 1. 拉取 W&B 数据

```bash
.venv/bin/python alignment/pull_wandb_metrics.py \
  --entity wncfht --project One-Shot-OPD \
  --run-id mdxt04lm be00rj9e --out alignment/runs
```

这个步骤只做外部数据落盘，不计算 alignment 指标。已有文件默认不会重复拉取；需要更新时使用 `--force`。

这些 run 是离线 `wandb sync` 上来的：任务在集群跑完后才把日志同步到 W&B，所以 `run.state`（以及 meta.json 里的 `state`）可能是 `failed`/`crashed`，即使训练本身正常完成或被抢占后已续跑。**不要**用 `state` 判断一个 run 是否失败或数据是否完整；以实际拉到的 history 步数为准。

### 2. 计算标准 alignment 指标

```bash
.venv/bin/python alignment/compute_alignment.py \
  --runs-dir alignment/runs --out-dir alignment/derived --t-start 30
```

输出 `derived/<run_id>__alignment.csv` 和 `derived/summary.json`。入口脚本负责遍历输入和写文件，真正的计算在 `core/metrics.py`。

### 3. 画基础 alignment 图

```bash
.venv/bin/python -m alignment.pipelines.alignment.main \
  --derived-dir alignment/derived --figures-dir alignment/figures
```

基础图会保留 step 0 到末尾。step 30 以前使用浅色，step 30 用竖线标注为分析 baseline / clipping boundary。

### 4. 尾部拟合

```bash
.venv/bin/python alignment/fit_tail.py --derived-dir alignment/derived --plot
```

会比较幂律、拉伸指数和 floor+指数，并写入 `derived/tail_fits.json`。

### 5. lr-ablation（M4）、论文和 Qwen3 分析

lr-ablation 图共享 `config/runs.py` 的 M4 run 配置，并通过 `core.data` 和 `core.metrics` 取得数据和 distance：

```bash
.venv/bin/python -m alignment.pipelines.lr_ablation.figure19
.venv/bin/python -m alignment.pipelines.lr_ablation.supplement
.venv/bin/python -m alignment.pipelines.lr_ablation.panels
```

论文曲线和 Qwen3 诊断有独立数据源，不为了形式统一而强行塞进同一种 history schema：

```text
论文 PDF → extract_paper_figures.py → paper_curves/*.csv
Qwen3 eval JSONL + W&B history → pipelines/qwen3/main.py
```

它们可以共享 tail fitting、absorption 和 plotting style，但保留各自 loader 更清楚。

## 修改代码时的约定

1. 新数据源先加 loader，不要在绘图脚本里直接读取原始文件。
2. 新指标先加到 `core/metrics.py`，返回数组、DataFrame 或 JSON-compatible 结果。
3. 绘图函数接收已经计算好的数据，绘图层负责 baseline、坐标轴、颜色和输出路径。
4. 不要在多个脚本重复定义 run id、label、color 和 scale；M4 配置放在 `config/runs.py`。
5. 保留现有脚本作为可执行入口，但新功能应避免 import 时自动画图。
6. step 轴上的图一律用 `core.plotting` 的 `plot_series`（step 0 起、baseline 前淡化）和 `mark_baseline`（baseline 阴影加竖线），不要手写 `step >= t0` 截断或重复的淡化逻辑。
7. 指标口径发生变化时，先在 docstring 和 README 说明，不要默默改变已有 derived 文件的含义。

## 当前边界

`alignment/` 是离线分析工具，不负责 GPU 训练、远程集群提交或 checkpoint 修改。`spectral_effopd.py` 是 trainer-agnostic 的算法原型，和离线绘图 pipeline 共享研究背景，但不应被误认为是数据分析入口。
