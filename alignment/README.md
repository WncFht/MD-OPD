# 对齐度量：d_t（距离）与 v_t（吸收率）

OPD 教师-学生对齐的离线分析，度量定义来自 Fu et al. (2026) Eq. 6/7 与
`docs/2026-09-06-absorption-rate-spectral-theory.md`：

$$
d_t=\frac{1}{|\mathcal T_t|}\sum_{i\in\mathcal T_t}\left|\log\pi_T(y_i\mid s_i)-\log\pi_{\theta_t}(y_i\mid s_i)\right|,\qquad
v_t=\frac{d_t-d_{t+1}}{d_t}.
$$

度量从 step 30 开始追踪（此前 gradient clipping 仍在生效）；$v_t$ 另有论文的几何窗口 $[t/1.4,\,1.4t]$ 估计（见下）。

## 流水线

```
pull_wandb_metrics.py   W&B -> alignment/runs/<run_id>__history.csv + __meta.json
compute_alignment.py    CSV  -> alignment/derived/<run_id>__alignment.csv + summary.json
plot_alignment.py       CSV  -> alignment/figures/*.png
fit_tail.py             CSV  -> D1 尾部形态拟合（幂律 / 拉伸指数 / floor+指数）
```

拉取本仓库标准的两个 run（top16 vs sampled-token）：

```bash
.venv/bin/python alignment/pull_wandb_metrics.py \
    --entity wncfht --project One-Shot-OPD \
    --run-id be00rj9e mdxt04lm
.venv/bin/python alignment/compute_alignment.py
.venv/bin/python alignment/plot_alignment.py
.venv/bin/python alignment/fit_tail.py
```

`pull_wandb_metrics.py` 与项目无关：任意 `--entity/--project/--run-id` 均可；
输出文件已存在时自动跳过（`--force` 强制重拉）；默认键列表是 curated 的 OPD 指标集 `DEFAULT_KEYS`（可用 `--keys`/`--keys-file` 覆盖）。凭证来自 `~/.netrc` / `$WANDB_API_KEY`。

## 距离代理是什么

训练日志里没有直接的 $d_t$，但两个 run 都是 `adv_estimator=token_reward_direct`，trainer 记录的 `opd/token_reward_mean` 是逐 token reward $\mathrm{rm\_scores}=\log\pi_T-\log\pi_\theta$ 在访问 token 上的均值（见 `verl/verl/workers/fsdp_teacher_worker.py`）。我们取

$$d_t^{\mathrm{proxy}} = -\,\texttt{opd/token\_reward\_mean}$$

作为距离（nats/token）：

- **sampled-token** run（`mdxt04lm`，`log_prob_top_k=0`）：完整的 reverse-KL
  k1 估计，$d^{\mathrm{proxy}}_t$ 即每 token 平均 reverse KL。
- **top16** run（`be00rj9e`，`log_prob_top_k=16`，策略 `only_stu`）：同一量的
  top-16 截断版（也正是实际训练信号）。

注意：该代理是 KL 型（平方型）距离，不是论文的 L1 均值 $|\cdot|$，且 top16 截断会压低绝对水平，因此**两个 run 的绝对值不可横向比较，读的是衰减形状**（$v_t$、指数、比值）。

## v_t 的几何窗口估计

单步定义 $v_t=(d_t-d_{t+1})/d_t$ 在后期噪声太大。若在窗口内 $\ln d_s$ 近似线性：$\ln d_s \approx a + \beta s$，则 $d_{t+1}/d_t \approx e^{\beta}$，于是单步吸收率为

$$
\boxed{\;v_t^{\mathrm{geo}} \;=\; 1-\exp\!\big(\hat\beta_t\big),\qquad
\hat\beta_t=\underset{\beta}{\arg\min}\sum_{s\in[t/1.4,\;1.4t]}
\big(\ln d_s-a-\beta s\big)^2\;}
$$

即 $\hat\beta_t$ 是 $\ln d_s$ 对步数 $s$ 在几何窗口 $[t/1.4,\,1.4t]$ 上的 OLS 斜率。窗口点数 $\approx 0.686t$ 随 $t$ 增长，后期自然更平滑；窗口必须完整落在 $[t_{\mathrm{start}}, T]$ 内，所以估计只存在于 $t\in[1.4\,t_{\mathrm{start}},\,T/1.4]$——279 步的 run 即 $t\in[42,199]$，这正是论文脚注 3 里"$v_t$ 在 step 300 之前停止"的原因。

## derived 列含义（`alignment/derived/<run_id>__alignment.csv`）

| 列         | 含义                                                              |
| ---------- | ----------------------------------------------------------------- |
| `step`     | 训练步（`_step`）                                                 |
| `d_nats`   | 距离代理，nats/token                                              |
| `d_rel`    | $d_t/d_{30}$（相对 step-30 的剩余距离）                           |
| `v_direct` | 原始单步比值 $(d_t-d_{t+1})/d_t$（噪声大）                        |
| `v_geo`    | 几何窗口估计 $1-\exp(\hat\beta_t)$，窗口 $[t/1.4,\,1.4t]$，见上式 |

`summary.json` 是头部诊断：`d_end/d_start`、step 50/200 处的 `v_geo`、`v50_over_v200`（纯幂律预测 ≈ 4）、log-log 尾部幂律指数 `alpha`。

## D1 尾部形态拟合（`fit_tail.py`）

对 step ≥ 30 的 $d_t$ 分别拟合三种衰减形式并比较对数空间残差（笔记 §5 D1；每个模型在固定形状参数后退化为普通最小二乘，无需 scipy）：

| 模型            | 形式                          | 谱含义（笔记 §2.4）      |
| --------------- | ----------------------------- | ------------------------ |
| `power_law`     | $d=Ct^{-\alpha}$              | 幂律谱 + 振幅沿谱衰减    |
| `stretched_exp` | $d=Ce^{-(t/\tau)^\gamma}$     | 宽谱近均匀振幅（玻璃态） |
| `floor_exp`     | $d=d_\infty+Ae^{-t/\tau}$     | 单一主模式 + 噪声底      |

输出：`derived/tail_fits.json`（参数 + 各模型 rmse + 最优模型）、`derived/<run_id>__local_alpha.csv`（滑窗局部指数 $\alpha(t)$）、`figures/tail_fits__<run_id>.png`（`--plot`）。注意：$\gamma\lesssim0.2$ 时拉伸指数退化为"指数漂移的幂律"，网格在该处截断并在 JSON 里以 `at_grid_boundary` 标记。

## 图（`alignment/figures/`）

- `fig9_style.png` -- 论文 Figure 9 版式：左 $d_t/d_{30}$、右 $v_t$（%），对数纵轴 vs 线性步轴，双 run 对比。
- `distance_raw.png` -- 原始距离代理（nats/token）。
- `absorption_estimators.png` -- 每个 run 的单步 `v_direct` vs 几何窗口 `v_geo` 对比。
- `tail_fits__<run_id>.png` -- D1 三种尾部形态的拟合对比（log-log）。
