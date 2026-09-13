# MD-OPD / NGD 方向评估与下一步实验建议

> 结论：MD-OPD 与 NGD 不是两条路线，而是同一个目标更新在三个实现层次上的投影。已跑的 MDPO 数据显示它按外步最快到 floor、熵保持最好，但谱反演给出的效率上限约为 2–3×（到 floor 为止），不是数量级，且碰不到噪声 floor 与表征 gap。该方向的正确用途是充当谱理论的证伪仪器，外加一个仍空着的论文槽位。下一步先跑三个低成本的判决性读数实验（E1–E3），再决定要不要写 trainer 代码。

## 1. 实验盘点

### 1.1 已跑的四条线

| 线           | 代表 run                                                                               | 关键读数                                                                                         |
| ------------ | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 全量基线     | `mdxt04lm`（vanilla ratio，lr 1e-6，DAPO-math）                                        | d 从 ~0.10/token 降到 ~0.004，$d_{279}/d_{30}\approx0.061$，val best 69.55%                      |
| M4 lr 扫描   | `ybi7q89v` 5e-7、`yj2w13fq` 2e-6、`ml6mz58f` 5e-6、`l0a6rsc1` 1e-5、`tjjxnvw9` β2=0.95 | floor 深度随 lr 单调加深至 ~5e-6 饱和；1e-4（`vmk256kl`）发散剔除                                |
| one-shot t22 | `5aub1k4h` online 1-epoch；`ceejzu26` frozen+noratio；`0i38mr1k` frozen+vanilla ratio  | frozen+noratio 吸收最深（$d_{end}/d_{30}=0.012$、熵稳）；frozen+ratio 崩盘（d≈0.50，failed@218） |
| MD-OPD 系列  | `nh3jj9bt` 4 inner epochs；`zyskh7f2` 16 epochs；`6p579hnn` MDPO                       | 见下表                                                                                           |
| 跨族对照     | `um5vzc6m` Qwen3-1.7B→Qwen3-4B                                                         | gap 平在 0.14/token 共 158 步（拟合幂律 α=0.0096，约等于不动）                                   |

### 1.2 MD-OPD 系列读数（one-shot 配方，同一 rollout）

| run        | 机制                                        | $d_{last}$ | entropy 首→末   | val best |
| ---------- | ------------------------------------------- | ---------- | --------------- | -------- |
| `5aub1k4h` | 1 inner epoch                               | 0.00154    | 0.772→0.598     | 66.9%    |
| `nh3jj9bt` | 4 inner epochs                              | 0.00075    | 0.768→0.587     | 68.4%    |
| `zyskh7f2` | 16 inner epochs                             | 0.00409    | 0.761→0.629     | 69.2%    |
| `6p579hnn` | MDPO：4 epochs + old-policy k3 KL，η 1→0.01 | 0.00109    | 0.761→**0.632** | 69.1%    |

（`6p579hnn` wandb 状态 failed@299，但 299 步数据完整；来源 `plots/derived/md_opd_summary.json`）

三个观察：

1. **折到内层梯度步轴上，四个 arm 在 ~1000–1200 个梯度步内都碰到 ~1e-3 的 floor**。按外步数 MDPO 最快，但每梯度步的效率增益是边际的——多花的内层步数基本吃掉了省下的外层步数。
2. **16-epoch arm 停在 ~5e-3 反而更差**：陈旧 surrogate 饱和。内层在一个已经走掉的分布上继续过优化，超出某个点后不再转化为真实的 KL 下降。
3. **MDPO 是唯一 300 步后熵仍站在 0.63 的 arm**，且 $v_t$ 在 step ~200 后趋零并微负——这正是"已吸收完可吸收分量、顶到 floor"的预言行为，而不是失败。val 差距（68.4–69.2）在单 seed 噪声内。

## 2. MD-OPD 与 NGD 的关系：三层等价

"做 MD-OPD 还是 NGD"是假二选一。两者在三个实现层次上是同一个更新（推导见 `docs/MD-NGD/`，此处只列结论）：

- **A. 分布空间精确 MD**：单纯形上的 KL mirror descent 给出几何混合 $p^+ \propto \mu^{1-\eta}q^{\eta}$，由 Raskutti–Mukherjee 定理[^rm13]，它就是对偶（自然参数）坐标上的 NGD。OPD 的 sampled-token advantage $\log q - \log p$ 恰好就是指数梯度所消费的信号——**现有 reward 构造天然就是 EG 的输入**。
- **B. 参数空间局部二次**：任务线性化 + KL 二次化 + 精确求解，得到一步参数空间自然梯度 $-\eta F^{\dagger}g$。
- **C. 实际实现**：full OPD loss + anchor KL 用 $m$ 步 Adam/SGD 求解，是 KL-proximal-point 的不精确解，等价于谱滤波 $\delta_m=-[I-(I-\alpha B)^m]B^{-1}g$，$B=H+F/\eta$。$m=1$ 时退回 vanilla PG（anchor KL 在锚点梯度为零）；$m\to\infty$ 收敛到 $-(H+F/\eta)^{\dagger}g$，仅当 $H\approx F$ 时才是 $\approx-\frac{\eta}{1+\eta}F^{\dagger}g$。

一个实现语义事实：vanilla+ratio 的 surrogate 对 $p$ 是线性的（`core_algos.py:1286`），所以它的精确 prox 不动点**确实**是限制在采样 token 上的几何混合——MDPO 内层若收敛即实现分布空间 NGD；4 步 Adam 只是部分实现（截断的谱滤波）。

## 3. 效率上限：谱说了算

对 `ours_sampled_token_kl` 的 NNLS 反演谱（40 个 support modes，单步吸收率 $\rho\in[0.004,0.1]$，初始加权速率 $v_0\approx0.0274$/step，见 `plots/derived/inversion_summary.json`、`spectrum__ours_sampled_token_kl.csv`）做理想白化模拟——所有 mode 被预条件拉到同一速率 2.74%/step：

| 目标           | 理想白化所需步数 | 实际观测步数 | 倍数 |
| -------------- | ---------------- | ------------ | ---- |
| $d/d_{30}=0.2$ | 89               | ~169         | 1.9× |
| $d/d_{30}=0.1$ | 115              | ~296         | 2.6× |

且 $t=300$ 处的残余 100% 落在最慢三分位的 mode 上。含义：**任何形式的预条件/白化/速率均衡，到 floor 为止的收益上限约 2–3×**。这与 EffOPD[^effopd]声称的 ~3× 步数加速量级一致（他们用 checkpoint 外推 $W+2^k\Delta$ 走的是另一条路：利用方向可预测性，与我们正交）。

上限之外有两堵墙，预条件都碰不到：

- **噪声 floor**：$d_\infty^{KL}=(\eta/4B)\sum_i\sigma_i^2$。白化不改信噪比，`mdxt04lm` 的 floor_exp 拟合给出 $d_{inf}\approx0.001$（`tail_fits.json`）。
- **表征 gap**：`um5vzc6m` 的 0.14/token 平线。跨族师生对之间不可对齐的分量，任何优化器都压不掉。

还有一个容易被低估的点：**Adam 本身就是对角经验 Fisher 的部分白化器**[^fadam]，其低 SNR 行为已有反例[^muon]。M4 的 Adam 数据里可能已经包含了一半的"NGD 效应"——这与 MDPO 增益边际化是自洽的。

## 4. 这个方向有什么用

1. **谱理论的证伪仪器**。模型预言 $v_t=\eta\langle\lambda\rangle_t$ 单调下降（Proschan 递减失效率）且 floor = noise + gap。MDPO 是目前唯一改变了 $v_t$ 形态的干预臂：`6p579hnn` 的 $v_t\to0^-$ 与熵保持是对模型的定性确认，也给出了"何时停"的可观测信号。
2. **熵保持的机制证据**。几何混合 $p^+\propto\mu^{1-\eta}q^\eta$ 保 support（$p_i=0$ 处不增质量），与 `0i38mr1k`（vanilla ratio 崩盘）形成机制对照。pass@K 回退类批评[^ida]意味着"更快到 floor"必须在能力天花板上度量，而不是只看 KL。
3. **论文槽位**。文献核查（~35 篇逐一对 arXiv 原文验证）的结论：
    - 已被占：更新形式 $\pi^+\propto\pi_{old}e^{\eta A}$（REPS→MPO→PMD→OPMD[^mdpo][^kimi]）；LM 上的几何混合更新（Nash-MD[^nashmd]、IPO-MD 等偏好对齐一族）；"KL-to-old on LLM"（OPMD、PMD-mean[^pmdmean]、BOLT、ProRL 移动锚）。
    - 最危险的近邻：**GeoSD**[^geosd] 已发"K-FAC 参数空间 NGD + 移动 Fisher 锚"用于 on-policy **自**蒸馏——我们的主张必须收窄到 teacher-student OPD + 输出空间故事。
    - 仍空的：①"centered-logit 回归 + 普通 GD = reverse-KL 输出空间 NGD"这一 identity 无人 claim——CSD[^csd]、BiLD、logit standardization 做了 loss 没点破几何；②首个 teacher-OPD 的 MD/NGD formulation 与 trajectory/fixed-point 分析（mode-seeking 路径、support 保持、$\eta=1$ 一步到位、隐式 KL–χ² 正则）。
    - 定位判断：mechanism 本身是 folklore，论文必须卖**分析与读数**，不能卖更新形式；否则就是 "MPO-for-OPD rebranding"。

## 5. 下一步实验（按判决力/成本排序）

原则沿用 brief 的规则：先修读数，再谈干预。E1–E3 都不需要新 trainer 代码。

**E1. 冻结平台 lr 退火探针（几乎零成本）**。取 frozen 配方跑到平台后 lr ×0.1 续跑。floor 随之下降 → 噪声 floor（$\propto\eta\sigma^2/4B$）坐实，方差工程与白化有意义；不降 → $d_{gap}$/表征分量主导，预条件帮不上。**先看 M4 各 lr arm 的 floor 深度对比（5e-7 vs 1e-6 vs 5e-6 vs 1e-5），答案可能已在数据里**，只需画图。

**E2. 逐 token 频率十分位 × 符号分解残余（即 M3）**。量出剩余 $d$ 中有多少位于 $\lambda<\lambda_{crit}$（不可吸收，NGD 够不着）vs 以上（NGD 的可服务市场）。顺手解决 CE−H 指标伪影问题。

**E3. 单步 CLR 手术探针（决定性）**。取 frozen checkpoint，用现有 top-16 logits 构造 head+OTHER 的 CLR loss $L=\frac12\|P(z-u)\|^2$，分别走 SGD 与 Adam **一步**，量 $\Delta d$/步。在写任何 trainer 代码之前，直接测"自然化方向每步多吸收多少"。这是整个 NGD 方向的 go/no-go。

之后按 E3 结果分叉：

- **E4**：head+OTHER CLR + **SGD** 干净 arm（真 CLR run；VGD arm 作对照）。注意必须含 SGD 臂，否则无法把"CLR 方向"与"Adam 自带的对角白化"分开。
- **E5**：MDPO η 方向消融——常数 η 与递增 η 两臂。当前 1→0.01（kl_coef 1→100）等价于信任域随训练收紧；homotopy/annealed-PMD 文献[^homotopic]的标准方向是**衰减**正则，现在的方向可能是后期停滞/距离微升的原因之一。
- **E6**：M5 Lanczos 直接测 frozen-state Hessian 谱，验证 NNLS 反演。
- **E7**（第三备选路线）：`plots/spectral_effopd.py` 低秩谱外推（checkpoint delta 的 SVD 基 + Rayleigh 曲率 + modal 外推），与 MD/NGD 正交，尚未接入 verl。

**明确不做**：E3 出结果前不写 K-FAC/trust-region 机器——GeoSD 已占参数空间 NGD 自蒸馏，且上限分析说明奖品 ≤3×；不再用跨族对（Qwen3）问算法问题——$d_{gap}$ 主导时测不出优化器差异。

## 6. 总判断

"做 MD-OPD 或 NGD 对吗"——作为"换一个优化目标来实现同一个几何更新"：对，且已被 `6p579hnn` 部分验证（最快到 floor + 熵保持最好）。作为"拿数量级加速"：不对，实测谱给出的上限只有 2–3×，且 floor 与 gap 都碰不到。它的真实价值有二：谱吸收模型的第一个可控干预/证伪工具；以及一个还没被占的论文槽位（CLR=NGD identity + OPD 专属 trajectory 分析）。下一步不是再跑一个 MD 变体，而是 E1–E3 三个读数实验，用最低成本把整个干预方向的 go/no-go 定下来。

### 参考文献

[^rm13]: G. Raskutti, S. Mukherjee. The Information Geometry of Mirror Descent. arXiv 2013. [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)

[^mdpo]: M. Tomar, L. Shani, Y. Efroni, M. Ghavamzadeh. Mirror Descent Policy Optimization. ICLR 2022. [arXiv:2005.09814](https://arxiv.org/abs/2005.09814)

[^kimi]: Kimi Team. Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv 2025. [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)（OPMD：KL-to-old-policy mirror descent 用于 LLM RLVR）

[^nashmd]: R. Munos et al. Nash Learning from Human Feedback. arXiv 2023. [arXiv:2312.00886](https://arxiv.org/abs/2312.00886)（LM 上的几何混合更新 $\pi^{1-\eta\tau}\pi_{ref}^{\eta\tau}e^{\eta P}$）

[^pmdmean]: Y. Xu et al. Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training. arXiv 2026. [arXiv:2602.05933](https://arxiv.org/abs/2602.05933)

[^geosd]: I. Jukić, I. Titov. Geometric Self-Distillation for Reasoning Generalization. arXiv 2026. [arXiv:2607.06855](https://arxiv.org/abs/2607.06855)（K-FAC 经验 Fisher NGD + 每 64 步刷新的 Fisher–Rao 移动锚，Hellinger 损失，自蒸馏）

[^csd]: Kim et al. Distillation of LLMs via Concrete Score Matching. ICLR 2026. [arXiv:2509.25837](https://arxiv.org/abs/2509.25837)（成对 logit 差匹配，无 NGD 声明）

[^effopd]: Y. Cai et al. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^fadam]: FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information. arXiv 2024. [arXiv:2405.12807](https://arxiv.org/abs/2405.12807)

[^muon]: Whitening/preconditioning 在低 SNR 失效的分析。arXiv 2026. [arXiv:2605.19282](https://arxiv.org/abs/2605.19282)

[^homotopic]: X. Li, G. Lan, T. Zhao. Homotopic Policy Mirror Descent. arXiv 2022. [arXiv:2201.09457](https://arxiv.org/abs/2201.09457)（衰减正则的 PMD）

[^ida]: IDA-OPD：pass@k 多样性坍缩诊断与 entropy-influence advantage shaping. arXiv 2026. [arXiv:2608.29846](https://arxiv.org/abs/2608.29846)
