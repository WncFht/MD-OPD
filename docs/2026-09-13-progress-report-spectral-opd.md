# OPD 吸收动力学：进度汇报（理论 + 实验 + 文献位置 + 计划）

> **30 秒版本**：On-policy distillation（OPD）这条线已经证明极少数据就够蒸馏——一篇近期的 OPD 论文用**一条训练样本**跑通了配方[^opd2]。当数据不再是瓶颈，剩下的问题就是"每步算力买到多少教师"——吸收效率。我们把这个过程建成 Fisher 谱上的并行指数衰减模型，从实际训练曲线里反演出谱（40–64 个 mode，单步吸收率横跨 25 倍），模型给出两个可证伪预言（吸收速率单调下降、floor = 噪声 + 表征 gap），并算出任意"白化/自然梯度"类干预的效率上限约 **2–3×**。MD-OPD 系列实验作为第一个干预，验证了预言的形态。下一步是三个近零成本的判决性实验，决定干预方向的 go/no-go。

---

## 1. 背景与 motivation

### 1.1 OPD 是什么

学生在**自己的 rollout 上**被教师逐 token 打分。每个采样位置 $s_i=(x,y_{<i})$、采样 token $y_i$ 上，奖励是

$$
r_i=\log \pi_T(y_i\mid s_i)-\log \pi_\theta(y_i\mid s_i),
$$

用普通 policy gradient 消费。期望意义下这等价于在学生诱导的状态分布上最小化逐 token 反向 KL：$J(\theta)=\mathbb E_s\,\mathrm{KL}\bigl(\pi_\theta(\cdot|s)\,\|\,\pi_T(\cdot|s)\bigr)$。相对离线 SFT 蒸馏，它的监督落在学生实际会访问的状态上——这是它近两年成为标准后训练配方的理由[^minillm][^gkd][^tm][^opd1]。

可观测的训练曲线是 $d_t=-\overline{A}_t$（采样 token 的 KL 距离估计），以及吸收速率

$$
v_t=\frac{d_t-d_{t+1}}{d_t},
$$

即"这一步又吃掉了剩余距离的百分之几"。

### 1.2 为什么要研究动力学

三个理由，按重要性排：

1. **数据问题已被解决，算法问题裸露出来**。OPD-II 证明一条样本够；Rethinking OPD-I 的概括是 "data-overfed, algorithm-starved"[^opd1]。当数据不再是变量，优化器与目标的几何就是仅剩的杠杆。
2. **现象一堆、解释没有**。我们自己的实验里就有四个待解释事实（§3.2）：衰减先快后慢并停在非零 floor；同配方换个 loss 形式就崩盘；lr 加大只把 floor 压浅一点然后饱和；跨族师生对 158 步纹丝不动。社区文献也在堆积互相打架的观察（mode-seeking、熵坍缩、pass@k 退化、"其实只是抑制低概率 token"[^opsa]）。需要一个能统一解释的框架。
3. **干预需要地图**。想加速的人已经来了：EffOPD 用 checkpoint 外推声称 ~3×[^effopd]，GeoSD 上 K-FAC 自然梯度[^geosd]。没有理论就只能跟着试错；有了谱模型，每种干预的收益上限**可以先算出来再决定要不要跑**。

### 1.3 核心问题

> 逐 token 的师生 KL 距离是按什么动力学被吸收的？floor 由什么构成？哪些杠杆能移动它、各自上限多少？

## 2. 理论：谱吸收模型

完整推导在 `docs/2026-09-07-opd-efficiency-theory-brief.md` 与 `docs/MD-NGD/`，这里给出最小推理链。

### 2.1 推导链

**第 1 步：冻结状态 ⇒ 固定目标。** 一个 outer step 内 rollout 由旧策略生成，状态分布 $d_\mu$ 冻结。于是在这一步内，OPD 就是一个固定目标

$$
J(\theta)=\mathbb E_{s\sim d_\mu}\mathrm{KL}\bigl(\pi_\theta(\cdot|s)\,\|\,\pi_T(\cdot|s)\bigr)
$$

上的确定性优化问题。on-policy 的"目标随策略漂移"被逐步冻结切成了分段静态问题。

**第 2 步：局部二次 ⇒ Hessian = Fisher。** 在当前点附近把 $J$ 二次化，主导项是 logits 经 softmax 的逐位置 Fisher：$A=\mathbb E_s[J_s^\top C(\pi_\theta)J_s]$，其中 $C(p)=\mathrm{Diag}(p)-pp^\top$（Cai et al. 附录 F.5 的同一个 $A$[^effopd]；我们文档里推导了它的精确适用边界——远离教师时还有 residual-curvature 项，见 `docs/MD-NGD/OPD加KL是否等于一步NGD.md` §7）。

**第 3 步：GD 更新 ⇒ 谱上并行指数衰减。** 把 $A$ 对角化，步长 $\eta$ 的梯度步在每个特征方向上独立收缩：

$$
x_i(t+1)=(1-\eta\lambda_i)\,x_i(t)
\quad\Longrightarrow\quad
d(t)\approx\sum_i a_i\,e^{-\eta\lambda_i t}.
$$

**距离曲线是谱的拉普拉斯变换**——这一条是整个框架的枢纽：可观测量（$d_t$、$v_t$）和几何对象（Fisher 谱）之间有了可逆变换，实测曲线可以反演出谱（我们用 NNLS 做了，`plots/derived/spectrum__*.csv`）。

**第 4 步：两个可证伪预言。**

- **预言 A（速率递减）**：$v_t=\eta\langle\lambda\rangle_t$ 是对残余质量的加权平均速率；快 mode 吸完后权重向慢 mode 迁移，所以 $v_t$ **单调下降**（Proschan 递减失效率的推论）。不依赖任何拟合，是谱模型的形状签名。
- **预言 B（floor 分解）**：SGD 噪声给每个 mode 一个稳态方差，不可吸收的小信号沉底：

$$
d_\infty \;=\; \underbrace{\frac{\eta}{4B}\sum_i\sigma_i^2}_{\text{噪声 floor}}
\;+\;\underbrace{d_{gap}}_{\text{表征/可达性差}},
\qquad
\lambda_{crit}\sim\frac{\sigma_i}{c_i}\sqrt{\frac{\eta}{2B}} .
$$

floor 不是"没训练够"，是**噪声与不可达分量的叠加**——不同成因对应不同干预（降噪 vs 换师生对），这就是 E1 实验要分辨的事。

**第 5 步：干预地图。** 每种干预在谱语言里都是一个算子：

| 干预                                   | 谱语言                                                  | 上限来源                        |
| -------------------------------------- | ------------------------------------------------------- | ------------------------------- |
| 预条件 / NGD / 白化                    | 把所有 mode 拉到同一速率 $\bar\lambda$                  | 只省"快慢不均"，碰到 floor 为止 |
| 方差工程（batch、baseline、estimator） | 降 $\sigma_i^2$                                         | 直接压噪声 floor                |
| 学习率/吸收闭环                        | $\eta$ 调度 = 沿 $\langle\lambda\rangle_t$ 走的步长控制 | 稳定性 vs 地板噪声的权衡        |
| 谱外推（EffOPD 型）                    | 利用方向可预测性外插                                    | 与速率均衡正交                  |
| token 加权 / gating                    | 改写各 mode 的 $c_i,\sigma_i$                           | 信噪比整形                      |
| 换师生对                               | 改 $d_{gap}$                                            | 唯一碰 $d_{gap}$ 的杠杆         |

### 2.2 NGD / mirror descent 在这张地图里的位置

三层等价（推导见 `docs/MD-NGD/`）：

- **分布空间精确 MD**：KL mirror descent 在单纯形上的闭式解是几何混合 $p^+\propto\mu^{1-\eta}q^{\eta}$，由 Raskutti–Mukherjee 定理即对偶坐标上的 NGD[^rm]。OPD 的 advantage $\log q-\log p$ 恰好就是指数梯度吃的信号——现有 reward 构造天然是 EG 输入。
- **参数空间局部二次**：任务线性化 + KL 二次化 + 精确解 = $-\eta F^{\dagger}g$。
- **实际实现**：full OPD loss + anchor KL 跑 $m$ 步 Adam = KL-proximal 的不精确解，谱滤波 $\delta_m=-[I-(I-\alpha B)^m]B^{-1}g$；$m=1$ 退回 vanilla PG。

结论：白化/NGD/MD 全部只是"速率均衡"这一档干预，**上限约 2–3× 到 floor**（§4.3 的实测计算），且碰不到噪声 floor 和 $d_{gap}$。

## 3. 实验进度

### 3.1 run 盘点（21 个已拉取，`plots/runs/*__meta.json`）

| 线              | 代表 run                                                                                       | 读数                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 全量基线        | `mdxt04lm` vanilla ratio, lr 1e-6, DAPO-math                                                   | $d$ 0.10→0.004/token，$d_{279}/d_{30}\approx0.061$，val best 69.55%   |
| M4 lr 扫描      | `ybi7q89v`→`l0a6rsc1`（5e-7~1e-5）+ `tjjxnvw9` β2=0.95                                         | floor 随 lr 加深至 ~5e-6 饱和；1e-4 发散                              |
| one-shot t22    | `5aub1k4h`（online 1-epoch）；`ceejzu26`（frozen+noratio）；`0i38mr1k`（frozen+vanilla ratio） | 最深吸收 $d_{end}/d_{30}=0.012$；vanilla ratio 崩盘 d≈0.50 failed@218 |
| **MD-OPD 系列** | `nh3jj9bt` 4-epoch；`zyskh7f2` 16-epoch；`6p579hnn` MDPO                                       | 见 §4.2                                                               |
| 跨族对照        | `um5vzc6m` Qwen3-1.7B→4B                                                                       | gap 平在 0.14/token × 158 步（幂律 α=0.0096≈不动）                    |
| 其他            | `be00rj9e` topk16；`mzwm9q04`/`ur3j4vb5` discounted reward                                     | 指标口径与 reward 整形对照                                            |

### 3.2 四个待解释事实 → 模型怎么说

| 现象                                    | 谱模型解释                                                                                         | 状态                     |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| $d_t$ 先快后慢，$v_t$ 递减              | 快 mode 先吸完，残余质量向慢 mode 迁移（预言 A）                                                   | 与全部 run 形态一致      |
| 非零 floor ~1e-3                        | 噪声 floor + 不可达分量（预言 B），`mdxt04lm` floor_exp 拟合 $d_{inf}\approx0.001$                 | 拟合一致；成因待 E1 分辨 |
| frozen+vanilla ratio 崩盘（`0i38mr1k`） | ratio surrogate 的 clip/非线性在冻结 anchor 上放大离轴漂移；noratio 的 score-function 形式无此问题 | 定性解释成立             |
| 跨族对平线（`um5vzc6m`）                | $d_{gap}$ 主导：不可对齐分量吸不掉，$v_t\approx0$                                                  | 强证据                   |

### 3.3 谱测量（NNLS 反演，`plots/derived/`）

- `ours_sampled_token_kl`：40 个 support mode，$\rho\in[0.004,\,0.1]$/步（快慢差 25 倍），初始加权速率 $v_0\approx2.74\%$/step；
- `ours_top16_kl`：27 modes，形态相似；
- 尾部幂律指数 α≈1.31–1.55，与"模式数 >40 的连续谱"相容；
- holdout 检验：前 200 步拟合的谱外推后 50 步，log-RMSE 0.055，不差于幂律基线。

诚实的局限：反演是间接测量（把 $d_t$ 当 Laplace 变换反解），单 seed、单一 1.5B 同族对、optimizer 状态（Adam 二阶矩）会混入"谱"读数。所以排了 M5：用 Lanczos 在 frozen state 上直接测 Hessian 谱做交叉验证。

## 4. MD-OPD 支线：第一个干预实验

### 4.1 它是什么

在同一批 frozen rollout 上做 $m$ 个 inner epoch（任务项 + old-policy k3 KL 锚，η 从 1 线性退到 0.01，即 KL 系数 1→100）。理论上：内层若精确收敛，等价于分布空间 MD 步（几何混合）；有限 4 步 Adam 是截断逼近。

### 4.2 读数（`plots/derived/md_opd_summary.json`）

| run        | 机制          | $d_{last}$ | entropy 首→末   | val best |
| ---------- | ------------- | ---------- | --------------- | -------- |
| `5aub1k4h` | 1 inner epoch | 0.00154    | 0.772→0.598     | 66.9%    |
| `nh3jj9bt` | 4 epochs      | 0.00075    | 0.768→0.587     | 68.4%    |
| `zyskh7f2` | 16 epochs     | 0.00409    | 0.761→0.629     | 69.2%    |
| `6p579hnn` | MDPO          | 0.00109    | 0.761→**0.632** | 69.1%    |

三点：

1. **折到内层梯度步轴上，四个 arm 在 ~1000–1200 个梯度步内全部碰到 ~1e-3 floor**——MDPO 省的是外步，每梯度步增益边际。
2. **16-epoch 反而停在 ~5e-3**：陈旧 surrogate 饱和——在已经走掉的分布上过优化不再转化为真实 KL 下降。这直接证伪了"内层越多越好"。
3. **MDPO 熵保持全场最好（0.632）**，且 $v_t$ 在 ~200 步后趋零转负——按预言 B 这是"可吸收分量吸完、顶到 floor"的签名，不是失败。几何混合保 support（$p_i=0$ 恒死）的机制解释与读数自洽。

### 4.3 效率上限的实测计算

拿反演谱模拟"完美白化"（所有 mode 拉到同一 2.74%/步）：

| 目标           | 理想白化 | 实际    | 倍数 |
| -------------- | -------- | ------- | ---- |
| $d/d_{30}=0.2$ | 89 步    | ~169 步 | 1.9× |
| $d/d_{30}=0.1$ | 115 步   | ~296 步 | 2.6× |

$t=300$ 的残余 100% 落在最慢三分位。**结论：任何速率均衡类干预（NGD/K-FAC/MD 精确版/CLR）的收益上限 ≈2–3× 到 floor，且对 floor 本身无能为力。** 这与 EffOPD 声称的 ~3× 量级自洽——他们的外推走的是正交机制。另有脚注：Adam 本身就是对角经验 Fisher 的部分白化[^fadam]，我们数据里可能已含一半"NGD 效应"，与 MDPO 增益边际化自洽。

## 5. 文献位置（逐一核过 arXiv 原文，~35 篇）

**启发来源**：

- 动力学/几何：Cai et al. F.5 局部二次模型[^effopd]；Raskutti–Mukherjee 的 MD=对偶 NGD[^rm]；Kakade NPG→REPS→MPO→PMD 的指数梯度链[^mdpo]。
- 直接证据：GeoSD 证明 K-FAC NG + 移动 Fisher 锚在自蒸馏上可行且便宜（1.10× wall-clock）[^geosd]；EffOPD 证明 checkpoint 外推 ~3×[^effopd]。
- 反面警示：Adam≈对角 Fisher[^fadam]；低 SNR 下白化反而有害[^muon]；pass@k 坍缩批评要求"更快"必须在能力天花板上度量[^ida]。

**已被占（不能主张）**：更新形式 $\pi^+\propto\pi_{old}e^{\eta A}$（REPS→MPO→PMD→Kimi OPMD[^kimi]）；LM 上的几何混合更新（Nash-MD[^nashmd]）；KL-to-old on LLM（OPMD、PMD-mean[^pmdmean]、ProRL 移动锚）；**参数空间 K-FAC NGD + 移动 Fisher 锚的 on-policy 自蒸馏（GeoSD）**；Hellinger/α-族损失位（GeoSD）；MDPO 名字本身（Tomar 2020[^mdpo]，MuJoCo）。

**仍空（可主张）**：

1. **teacher-student OPD 的 MD/NGD 化**——没有人对独立教师蒸馏写过这个 formulation，附带 trajectory/fixed-point 分析（mode-seeking 路径、support 保持、$\eta=1$ 一步到教师、隐式 KL–χ² 正则[^pmdmean]）。
2. **"centered-logit 回归 + 普通 GD = reverse-KL 输出空间 NGD"这一 identity**——CSD[^csd]/BiLD/logit-standardization 做了同类 loss 但没点破几何（`docs/2026-09-09-centered-logit-regression-fisher-opd.md`）。
3. **谱吸收理论本身**——收敛速率、$v_t$、$\lambda_{crit}$、floor 分解、实测谱，文献里零覆盖。

定位判断：更新形式是 folklore，论文必须卖**理论 + 读数**；最危险近邻是 GeoSD（详细对比见 `docs/2026-09-13-geosd-vs-md-opd-ngd.md`）。

## 6. 下一步计划

沿用项目规则：**先修读数，再谈干预**。

| 优先级 | 实验                                                                             | 判决的问题                                                                                 | 成本                            |
| ------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------- |
| **E1** | 冻结平台 lr 退火（平台后 lr×0.1）                                                | floor 是噪声（降→方差工程有用）还是 gap（不降→预条件无用）                                 | ≈0，先看 M4 已有 floor 深度对比 |
| **E2** | 逐 token 频率十分位×符号分解残余（M3）                                           | 剩余 $d$ 里 $\lambda<\lambda_{crit}$ 不可吸收部分占多少 = NGD 的可服务市场                 | 低，纯读数                      |
| **E3** | 单步 CLR 手术探针：frozen ckpt 上 head+OTHER CLR loss，SGD/Adam 各一步，量 Δd/步 | "自然化方向"每步是否真多吸收——整个 NGD 方向的 go/no-go                                     | 低，不写 trainer                |
| E4     | head+OTHER CLR + SGD 干净 arm                                                    | E3 通过后的正式干预验证                                                                    | 中                              |
| E5     | MDPO η 方向消融（常数/递增 η，可加锚滞后 K∈{1,64,∞} 维）                         | 现 1→0.01 方向可能与 homotopy-PMD 标准方向相反[^homotopic]；GeoSD 的 K=64 提示锚滞后是空轴 | 中                              |
| E6     | M5 Lanczos 实测 frozen-state Hessian 谱                                          | 交叉验证 NNLS 反演                                                                         | 中                              |
| E7     | `plots/spectral_effopd.py` 谱外推路线                                            | 与 MD/NGD 正交的 ~3× 候选                                                                  | 中                              |

**明确不做**：E3 出结果前不写 K-FAC/trust-region 机器（GeoSD 已占位 + 上限 ≤3×）；不再用跨族对问算法问题（$d_{gap}$ 主导时测不出优化器差异）。

## 7. 汇报问答预案

- **"这跟 PPO-KL/TRPO 有什么区别？"** 机制上 MD-OPD 的 prox 与 TRPO 同族；我们的贡献不在更新形式而在：①把它放进了谱理论（每步吸收多少由谱决定，不是黑盒）；②指出分布空间精确解是几何混合，参数空间近似是截断谱滤波——这把"inner epoch 数"从超参变成了有预测行为的量（16-epoch 饱和就是证据）。
- **"为什么不直接上 K-FAC？"** 三个理由：GeoSD 已占参数空间 NG 自蒸馏；谱上限只有 2–3×；E3 用零成本先测自然化方向的单位步收益，go 了再写。
- **"2–3× 值得做吗？"** 如果只卖加速不值得；卖的是"OPD 动力学可计算、可证伪、干预收益可预算"这套理论，MD-OPD 是它的第一个验证仪器。另外天花板数字本身（包括"为什么不能 10×"）是没人写过的内容。
- **"谱反演可信吗？"** 是间接推断，所以 E6 用 Lanczos 直测交叉验证；holdout 外推误差已量化（log-RMSE 0.055）。
- **"val 提升是真的吗？"** 单 seed、差异 66.9–69.2% 在噪声内，不做主张；GeoSD 的 RevKL baseline 提示纯 KL 吸收可能买 OOD 代价，计划加 $d_{FR}(\pi_\theta,\pi_0)$ drift 读数与 OOD 口径。
- **"one-shot 是不是噱头？"** 恰恰相反，one-shot 是本框架的理想实验台：固定状态 ⇒ 固定目标 ⇒ 谱模型字面成立；全量数据里是分段近似的成立。

## 8. 一句话收束

我们把 OPD 从"一个配方"变成了"一个可计算的动力学对象"：距离曲线是 Fisher 谱的拉普拉斯变换，吸收速率、floor、干预上限全都从谱里读得出来；MD-OPD 系列实验确认了预言形态，也顺手量出了加速的天花板（2–3×）。接下来三个便宜实验决定：值得做的干预是"均衡速率"（NGD/CLR 路线）还是"压噪声 floor"（方差工程路线），抑或两样都不值得、把力气全部投给理论 + 空着的论文槽位。

### 参考文献

[^minillm]: Y. Gu et al. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)

[^gkd]: R. Agarwal et al. On-Policy Distillation of Language Models (GKD). ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

[^tm]: K. Lu et al. On-Policy Distillation. Thinking Machines, Oct 2025.

[^opd1]: Rethinking On-Policy Distillation of Large Language Models I. arXiv 2026. [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)

[^opd2]: Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^effopd]: Y. Cai et al. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^geosd]: I. Jukić, I. Titov. Geometric Self-Distillation for Reasoning Generalization. arXiv 2026. [arXiv:2607.06855](https://arxiv.org/abs/2607.06855)

[^rm]: G. Raskutti, S. Mukherjee. The Information Geometry of Mirror Descent. arXiv 2013. [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)

[^mdpo]: M. Tomar et al. Mirror Descent Policy Optimization. ICLR 2022. [arXiv:2005.09814](https://arxiv.org/abs/2005.09814)

[^kimi]: Kimi Team. Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv 2025. [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)

[^nashmd]: R. Munos et al. Nash Learning from Human Feedback. arXiv 2023. [arXiv:2312.00886](https://arxiv.org/abs/2312.00886)

[^pmdmean]: Y. Xu et al. Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training. arXiv 2026. [arXiv:2602.05933](https://arxiv.org/abs/2602.05933)

[^homotopic]: X. Li, G. Lan, T. Zhao. Homotopic Policy Mirror Descent. arXiv 2022. [arXiv:2201.09457](https://arxiv.org/abs/2201.09457)

[^csd]: Kim et al. Distillation of LLMs via Concrete Score Matching. ICLR 2026. [arXiv:2509.25837](https://arxiv.org/abs/2509.25837)

[^fadam]: FAdam: Adam as diagonal empirical-Fisher natural gradient. arXiv 2024. [arXiv:2405.12807](https://arxiv.org/abs/2405.12807)

[^muon]: Whitening/preconditioning under low SNR. arXiv 2026. [arXiv:2605.19282](https://arxiv.org/abs/2605.19282)

[^ida]: IDA-OPD: pass@k diversity collapse under OPD. arXiv 2026. [arXiv:2608.29846](https://arxiv.org/abs/2608.29846)

[^opsa]: Does OPD Really Distill? (OPSA). arXiv 2026. [arXiv:2608.31046](https://arxiv.org/abs/2608.31046)
