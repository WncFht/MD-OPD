# GeoSD（arXiv:2607.06855）与本项目的逐轴对比

> 一句话：GeoSD 与我们解决的是**不同的问题**，碰巧都用了"几何"这个词。他们问"密集蒸馏拉力会把学生拖到哪里去"（drift → OOD 退化），我们问"KL 距离按什么谱速率被吸收、floor 在哪"（吸收动力学 → 效率上限）。他们把几何用在 **θ 参数空间**（K-FAC 自然梯度 + Fisher–Rao 移动锚），我们把几何用在 **p 分布空间**（mirror descent / 几何混合 / 输出空间 NGD）。两篇工作在主张层面几乎不冲突，但 GeoSD 占掉了几个相邻槽位，写作时必须绕开或引用。

## 1. GeoSD 做了什么

论文：Jukić & Titov, _Geometric Self-Distillation for Reasoning Generalization_, arXiv:2607.06855（2026-07-07）[^geosd]。

**设定**：privileged-context **自**蒸馏。学生与教师是同一个网络：学生 $p_\theta^{(t)}=\pi_\theta(\cdot|x,y_{<t})$，教师 $q^{(t)}=\pi_\theta(\cdot|x,c,y_{<t})$，其中 $c$ 是一条离线验证过的正确解答（DAPO-Math 11k 题，每题至多保留 10 条 base 模型自采的正确 rollout，共 ~7k）。轨迹从当前学生 on-policy 采样（T=1.0，每步 8 题 × 1 rollout），教师 stop-gradient。

**三个组件**：

1. **Hellinger 损失**（弦距）：$D_H(p,q)=1-\sum_i\sqrt{p_i q_i}$。梯度权重 $w_i=\frac12\sqrt{p_iq_i}$——学生对某 token 没有质量时拉力自动消失，无论教师多自信。他们把它放进 α-divergence 族：α→±1 恢复前向/反向 KL，α=0 即 Hellinger，是自对偶不动点。
2. **参数空间自然梯度**：$\theta_{k+1}=\theta_k-\eta F_k^{-1}\nabla\mathcal L$，用 damped K-FAC **经验** Fisher 实现（14B/32B 用省显存变体）。开销：+27% optimizer FLOPs、+21% 显存、1.10× wall-clock。
3. **Fisher–Rao 近端项**（弧距，不是弦距）：$\lambda\cdot d_{FR}^2(\pi_\theta,\pi_{\theta_{ckpt}})$，锚是**每 64 步刷新一次的近端 checkpoint**——"移动信任域"，刻意不锚 $\pi_0$。在 NG 下该项变成"checkpoint pullback" $-2\eta\lambda F_k^{-1}F_{ckpt}\Delta_k$；$F_k\approx F_{ckpt}$ 时退化为普通 L2 拉拽。

**关键结果**：Qwen3 1.7B–32B + Olmo-3-7B-Think + DS-R1-Llama-8B，OOD（AIME24/25、AMC23、MATH-500）平均 +5.7~~8.6 分，同时保留 ID 增益。对照：FwdKL/RevKL 拿到最好 ID 但 OOD **退化** 3~~8 分；JSD 基本持平；TrOPD/TIP 过滤法只有小幅 OOD 增益。消融三件套全部 load-bearing：去 prox −6.7 OOD（drift 最大 0.32），Hellinger→JSD −4.4，NG→AdamW −3.8。

**他们的诊断**：教师对学生"还不能 justify"的 token 最自信，那里梯度最大；per-state 的界不等于路径的界，位移累积成 drift；高熵状态下 FwdKL/RevKL 造成 rank-1 collapse → "confident agreement on wrong answers"（假共识）。一句话定位：**"Selection alone cannot regulate update magnitude"**——过滤法管不了更新幅度。

## 2. 逐轴对比

| 轴                        | GeoSD                                                                                                          | 我们（One-Shot-OPD / MD-OPD / 谱理论）                                                                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 师生结构                  | **同一网络** + privileged context $c$                                                                          | **独立教师**（JustRL-DS-1.5B → R1-Distill-1.5B；另有跨族 Qwen3 对）                                                                                           |
| 蒸馏损失                  | squared Hellinger，$w_i=\frac12\sqrt{p_iq_i}$                                                                  | sampled-token reverse KL，advantage $\log q-\log p$                                                                                                           |
| 几何作用层                | **参数空间**：K-FAC NG + FR² 锚                                                                                | **分布空间**：几何混合 $p^+\propto\mu^{1-\eta}q^\eta$；CLR = 输出空间 NGD                                                                                     |
| 信任域                    | FR² 惩罚项加进损失，锚 = 64 步前的自己                                                                         | MDPO：KL(·‖old) 作 prox 算子，锚 = 上一步策略，η 1→0.01                                                                                                       |
| 要解决的问题              | drift / OOD 退化（信号幅度不可信）                                                                             | 吸收速率与 floor（优化器几何低效）                                                                                                                            |
| 理论内容                  | Čencov 定理、球面几何、近一致处各散度等价、checkpoint pullback 推导；**无谱/无收敛速率**，drift 论证是非正式的 | 谱吸收模型：$v_t=\eta\langle\lambda\rangle_t$、$d_\infty=(\eta/4B)\sum\sigma_i^2+d_{gap}$、$\lambda_{crit}$、NNLS 实测谱（40–64 modes，$\rho\in[0.004,0.1]$） |
| 对 reverse KL 的立场      | 视为基线之一，证明它在 privileged 设定下拿 OOD 换 ID                                                           | 它就是目标函数本身；分析它的 mode-seeking 轨迹与 floor                                                                                                        |
| mirror descent / 几何混合 | **全文未出现**（已核实）                                                                                       | 核心理论对象（`docs/MD-NGD/`）                                                                                                                                |

### 2.1 设定差异：gap 的位置不同

GeoSD 的教师 = 学生 + 上下文。教师分布与学生**同支撑、同参数化**，分歧全部来自 $c$ 携带的、学生推不出的信息。他们的问题不是"gap 能不能消掉"，而是"在低重叠态上该用多大力拉"——拉力本身要按重叠度 $\sqrt{pq}$ 衰减，路径长度要用 FR 锚拴住。

我们的教师是另一个模型。残余距离被谱模型分成三块：可吸收谱分量、噪声 floor、**表征 gap**（Qwen3 对 0.14/token 平线是后者主导的活标本）。GeoSD 的设定里几乎不存在我们意义的 $d_{gap}$；我们的设定里几乎不存在他们意义的"同一网络被自己带偏"。两种"吸不掉的质量"成分不同：他们的是**设计上要绕开的**（$c$ 的不可推断部分），我们的是**测量到的障碍**。

### 2.2 损失差异：权重函数的形状

三个逐 token 梯度权重并排看（都写成 $-\sum_i w_i\nabla\log p_i$）：

- FwdKL：$w_i=q_i$——$p_i\to0$ 处拉力反而最大；
- RevKL（我们）：$w_i=p_i\bigl(\log(q_i/p_i)+KL(p\|q)\bigr)$——零均值再分配，mode-seeking；
- Hellinger（GeoSD）：$w_i=\frac12\sqrt{p_iq_i}$——低重叠处拉力渐近消失。

注意我们 MD-OPD 的闭式更新 $p^+\propto\mu^{1-\eta}q^\eta$ 有一个他们没有的性质：**support 严格保持**——$p_i=0$ 的 token 永远拉不起来（指数梯度的乘法本性）。GeoSD 的 $\sqrt{pq}$ 是"趋零渐近"，EG 是"恒等于零"。这条性质归我们主张，且与熵保持读数（`6p579hnn` 熵 0.63 全场最高）互相印证。

### 2.3 几何作用的位置：θ vs p

这是最深的一层区别。

- **GeoSD**：目标函数是欧氏的（Hellinger 本身不含 Fisher 几何），几何全部由**优化器**引入——$F_k^{-1}$ 预条件 + 同几何的 prox 惩罚。他们的 NG 更新作用在参数 $\theta$ 上。
- **我们**：目标函数 reverse KL 的"正确几何"在分布空间——精确 MD 步是逐 token 的几何混合，等价于对偶坐标 NGD（Raskutti–Mukherjee）[^rm]。参数空间我们刻意保持欧氏（Adam/SGD），CLR 是把输出空间 NGD 拆进损失函数的手段。

用一句话说：**他们把"球面"花在损失函数（弦距）和锚（弧距）上；我们把 Fisher 花在谱（每个 mode 的吸收速率）和更新的函数空间闭式上。** GeoSD 论文里没有任何谱/收敛速率内容——他们的 F 是预条件器，我们的 F 是动力学算符。

### 2.4 锚的语义：同一信任域族的不同 lag

近一致处 $d_{FR}^2$、$D_H$、双向 KL 都塌缩成同一个 Fisher 二次型 $\frac12\delta^\top F\delta$（他们的 App. A.3 与我们的推导一致）。所以 GeoSD 的 FR² 移动锚和我们的 MDPO KL-to-old **在二阶上是同一个东西**，差别只有两点：

1. **锚的滞后**：他们 K=64 步刷新，我们每个外层步刷新。把"锚滞后 K"看成一个连续旋钮：K=1 是我们的 MDPO，K=64 是 GeoSD，K=∞ 是固定 $\pi_0$ 参考（PPO-KL/REF 系）。这条轴目前没人系统扫过。
2. **进入更新的方式**：他们把 prox 当惩罚项加进损失再过同一个 $F^{-1}$（得到 checkpoint pullback $-2\eta\lambda F_k^{-1}F_{ckpt}\Delta$）；我们把 KL 当 prox 算子用内层 SGD 求解（谱滤波 $I-(I-\alpha B)^m$）。精确求解极限下，他们 = "对 损失 + 拴链 做一步 NG"，我们 = "分布空间 MD 步"，前者多了对 64 步前自己的持续回拉，后者锚随步重置、不阻止最终收敛到教师。

### 2.5 解决的问题不同 → 不可互相替代

- 他们证据链：密集拉力在低重叠态最大 → 路径漂移 → OOD 退化、假共识。药方：衰减拉力 + 拴住路径。**衡量标准是 OOD 准确率**。
- 我们证据链：Fisher 谱色散 → 各 mode 吸收速率差 25 倍 → $v_t$ 递减、floor 由噪声+gap 构成。药方（候选）：速率均衡/白化。**衡量标准是每步吸收的 KL 质量**。

GeoSD 的 AdamW→NG 消融（−3.8 OOD）证明参数空间 NG 在_他们的_问题上有效；我们的谱模拟证明白化在_我们的_问题上上限 ~2–3×。两个效应在不同轴上，互相不能当对方的证据。

## 3. 他们堵了什么，留了什么

**被占（不要再主张）**：

- "参数空间 K-FAC NGD + 移动 Fisher 锚，用于 on-policy 蒸馏"——已发表，虽然是自蒸馏版本。我们若走参数空间 NG 必须引用并区分。
- Hellinger / α-divergence 作为蒸馏损失的位置（α 族框架把 fwd/rev KL 收作极限，Hellinger 居中被他们占掉）。
- "per-state bound ≠ path bound / 过滤法管不了幅度"这一诊断叙事。
- privileged-context / hint 蒸馏这个设定本身（他们有完整文献链）。

**仍空（我们的主张空间）**：

- **teacher-student OPD**（独立模型教师）上的 MD/NGD 化——他们的方法整套依赖"教师=自己"（教师 stop-gradient 但同权重、锚是旧 checkpoint 合理因为本来就是同一模型）。
- **reverse-KL 目标 + 分布空间闭式更新** $p^+\propto\mu^{1-\eta}q^\eta$——全文无 mirror descent，已核实。
- **CLR + 普通 GD = reverse-KL 输出空间 NGD** 这一 identity——他们没有任何等价物（他们的 NGD 在 θ，不在 logits）。
- **谱吸收理论**：收敛速率、$v_t$、$\lambda_{crit}$、floor 分解、实测谱——他们零覆盖。
- **support 保持的乘法更新 vs 渐近衰减的加法权重**：EG 的 $p_i=0$ 恒死性质是我们的。
- **锚滞后 K 的扫描**：K∈{1,64,∞} 是个空的实验轴。

**他们顺手给我们的警示**：他们的 RevKL baseline（= 我们的目标函数）在 privileged 设定下 OOD 退 3–8 分。虽然设定不同（我们教师是真实强模型、信号更"可 justify"），这仍是"只优化 KL 吸收可能在买 OOD 代价"的外部证据——我们的"更快到 floor"叙事需要 OOD 口径的 val 来防守。

## 4. 可执行的启示

1. **锚刷新间隔加入消融轴**。E5（η 方向消融）可扩展为 η×K 两维：我们的逐步锚 vs 类 GeoSD 的 K=64 锚 vs 固定参考。便宜，因为只需改 `compute_mdpo_eta`/锚更新逻辑。
2. **加一个 drift 读数**：$d_{FR}(\pi_\theta,\pi_0)$ 或等价 KL，现有 eval pipeline 可算。我们的谱框架预言 drift 主要沿慢 mode 累积——GeoSD 的 pullback 项恰好是均匀拴住所有 mode 位移的项，**他们的 prox 有效性可以被我们的谱语言解释**，写作时这是卖点不是撞车。
3. **OOD 口径 val**：把 MATH-500/AMC/AIME 与训练分布的关系讲清楚，或加真正的 OOD split，防"吸收快但漂移大"的反驳。
4. **若 E3 之后走参数空间 NG**：K-FAC 的实测开销只有 1.10× wall-clock（8B），比预期便宜，GeoSD App. B.6 是参考实现；但优先级仍在 CLR/输出空间路线之后。
5. **support 保持写进论文**：几何混合的"死 token 恒死"性质 + `6p579hnn` 熵保持读数，是我们相对 GeoSD 软衰减权重的差异化主张。

## 5. 一张对照卡片

|            | GeoSD                                                | 我们                                                      |
| ---------- | ---------------------------------------------------- | --------------------------------------------------------- |
| 核心隐喻   | 蒸馏是球面上的弦与弧：拉力按重叠衰减，路径按弧长拴住 | 蒸馏是 Fisher 谱的并行指数衰减：每个 mode 有自己的半衰期  |
| 几何坐标   | $\theta$（K-FAC $F^{-1}$）                           | $p$（EG/几何混合）与谱（$F$ 的特征分解）                  |
| 锚         | 64 步前的自己（移动）                                | 上一步的自己（MDPO）/ 无（vanilla）                       |
| 敌人       | drift：低重叠态上的过度拉力                          | 谱色散：快 mode 吸完、慢 mode 拖着 + 噪声 floor           |
| 已验证规模 | 1.7B–32B，三个模型族                                 | 1.5B 同族对 + 跨族反例                                    |
| 我们没有的 | OOD 主指标、drift 读数、K-FAC 实现、假共识分析       | —                                                         |
| 他们没有的 | —                                                    | 谱理论、收敛速率、MD 闭式、CLR=NGD identity、support 保持 |

### 参考文献

[^geosd]: I. Jukić, I. Titov. Geometric Self-Distillation for Reasoning Generalization. arXiv 2026. [arXiv:2607.06855](https://arxiv.org/abs/2607.06855)

[^rm]: G. Raskutti, S. Mukherjee. The Information Geometry of Mirror Descent. arXiv 2013. [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)
