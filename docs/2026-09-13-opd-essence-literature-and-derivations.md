# OPD 的本质、深层数学背景与 Adam+OPD 的改进路线：文献综述 + 新推导

> **核心结论**（接续 `2026-09-13-progress-report-spectral-opd.md` 与 `2026-09-13-implemented-opd-dynamics-derivation.md`）：
>
> 1. **OPD 在数学上是一个"插值型 + 乘性噪声"的优化问题**：采样 token 估计量在 $p=q$ 处逐样本梯度恒为零，强增长条件（SGC）结构性成立；因此常数学习率的 SGD/Adam 在理论上**没有经典的加性噪声地板**，观测到的 $\sim10^{-3}$ 地板必须另有来源（移动状态分布、支撑 gap、精度失配）。这把地板从"训练不够"变成一个可诊断的量。
> 2. **Adam 在 OPD 中扮演三重角色**：(a) 坐标对角的 sign 型均衡器，把 token 频率造成的谱色散拉平（Kunstner 2024 的重尾类不平衡机制）；(b) 对角经验 Fisher 预条件，而 OPD 的经验 Fisher $\propto a^2$ 随 $d\to0$ 消失，于是 $\hat v$ 塌陷、有效步长在地板附近膨胀；(c) $\beta_2$ 的长记忆构成一个**隐式学习率衰减**——实测 $\beta_2=0.999$ 臂晚期有效步长只剩 0.17–0.2 倍，$\beta_2=0.95$ 臂地板低 1.4–1.5 倍，且两臂曲线在"Adam 时间" $u=\sum\alpha g/\sqrt{\hat v}$ 下重合。
> 3. **白化不是免费的**：Martens 的常数步长噪声定理给出地板 $\frac{\alpha}{4}\mathrm{tr}((B-\frac{\alpha}{2}H)^{-1}\Sigma_g)$；在乘性噪声下推出一个**维度阈值** $\alpha\gamma n_{\rm eff}/4<1$，全白化把该阈值收紧 $n_{\rm eff}$ 倍。用反演谱模拟，全均衡（到均值速率）省 3.7 倍步数，Muon 式半白化省 2.4 倍——半白化是噪声与加速之间更稳妥的折中。
> 4. **文献空位**（94 篇 OPD 论文逐一核对）：OPD 里的优化器选择（Adam vs SGD vs Muon）、Fisher 谱、学习率与地板理论全部为空；被占的是 NGD 自蒸馏（GeoSD）、Fisher 投影目标（FP-OPD）、权重谱分析（2606.13657、2606.07082）、sign 型 advantage（TV-OPD）、近端/信赖域 OPD（TOP-D、TrOPD）。
> 5. **改进 Adam+OPD 的排序**：先做零成本诊断（$\beta_2$ 扫描或 Adam 时间对齐、rollout 与训练侧对数概率差、噪声 - 曲率对齐比），再做两类干预——**优化器层**（$\hat v$ 下限或真 Fisher 对角、矩阵参数上的 Muon 式半白化）与**目标层**（输出空间自然化：CLR / 几何混合目标）。MDPO 的 $\eta$ 调度按 PMD 理论应该"递减到平台而非归零"，并考虑 lazy 锚。

## 0. 与前两份文档的接口

前两份文档已经建立：$d(t)=\sum c_k e^{-\rho_k t}$ 的谱吸收模型、NNLS 反演谱（64 modes，$\rho\in[0.004,0.1]$，质量 0.085，初始加权速率 0.0274/步）、Adam 两相律、MDPO 的谱滤波映射 $\delta_m=-[I-(I-\alpha B)^m]B^{-1}g$、地板三分量分解。本文回答三个尚未回答的问题：

- 这些结构背后**更深的数学定理**是什么，各自的精确适用边界在哪；
- 从这些定理能否推出前两份文档没有的**新性质**；
- "Adam + OPD" 这一默认配方在这些性质下的**本质角色与改进方向**。

记号沿用：学生 $p_\theta=\mathrm{softmax}(z)$，教师 $q$，采样 token 上的 advantage $a=\log q-\log p$，$d_t=-\overline{a}_t$，Fisher $F=\mathbb E_s[J_s^\top(\mathrm{diag}\,p-pp^\top)J_s]$，学习率 $\alpha$。实现事实：`compute_opd_token_reward`（`opd_trainer_mixin.py`）给 $a$，`compute_policy_loss_opd_noratio`（`opd_core_algos.py`）给 $-a\log p$；vanilla 臂用 clipped ratio；AdamW $\alpha=10^{-6}$、$\beta=(0.9,0.999)$。

## 1. 数学背景：七块定理及其在 OPD 上的落点

### 1.1 Mirror descent 与 natural gradient 的精确对偶

Raskutti–Mukherjee 定理 1[^raskutti]：严格凸势 $G$ 诱导的 Bregman 散度 $B_G$ 做近端项的 mirror descent 步，**精确等于**对偶坐标 $\mu=\nabla G(\theta)$ 上、以 $\nabla^2G^*$ 为度量的自然梯度步。证明只是一行链式法则：MD 的一阶条件是 $\nabla G(\theta^+)=\nabla G(\theta)-\alpha\nabla f$，即对偶坐标里的平移。

在单纯形上取 $G=$ 负熵，则 $B_G=\mathrm{KL}$，对偶坐标是 logits，对偶度量 $\nabla^2\log\sum e^{\mu_j}=\mathrm{diag}\,p-pp^\top$ 正是 categorical Fisher。于是 **KL-MD on simplex $\equiv$ logits 上的 NGD，全局精确成立**，闭式是指数梯度。对 OPD 目标 $\mathrm{KL}(p\Vert q)$，$\nabla_pJ=\log(p/q)+1$，代入得几何混合 $p^+\propto p^{1-\eta}q^{\eta}$，$\eta=1$ 一步到教师。Amari 的 e/m 对偶给同一结论：logits 是 e-平坦坐标，$p$ 是 m-平坦坐标，几何混合是 e-测地线[^amari98]。

**边界**：当决策变量是网络参数 $\theta$ 时，$\mathrm{KL}(\pi_\theta\Vert\pi_k)$ 是输出空间的 Bregman、不是 $\theta$ 空间的 Bregman，Raskutti 定理不适用；只有在 $\theta_k$ 处二次化（Hessian = pullback Fisher）才得到**参数空间的局部**一步 NGD。这两层等价（输出空间精确、参数空间局部）是本项目所有"MD = NGD"陈述的准确形式。

**lazy vs greedy**：Nesterov 的 dual averaging 把近端中心固定在原点（lazy），MD 每步把中心重置到当前点（greedy）[^mcmahan]。MDPO 内层若每步重置 KL 锚 = greedy；锚定 rollout 旧策略 = lazy。§3.7 说明为何 OPD 上 lazy 锚方向更优。

### 1.2 插值与强增长条件

Schmidt–Le Roux[^schmidt13] 与 Vaswani–Bach–Schmidt[^vaswani19] 的 SGC：$\mathbb E_i\Vert\nabla f_i(x)\Vert^2\le\rho\Vert\nabla f(x)\Vert^2$。在此条件下常数步长 SGD 在强凸（或 PL）下**线性收敛到极小点本身**，无噪声地板；Nesterov 加速也保留。Ma–Bassily–Belkin[^ma18] 给出插值下 mini-batch 的临界 batch size：$m\le m^*$ 线性扩展、$m>m^*$ 饱和。

Adam 侧：Zhang 等[^zhang22adam] 证明 $\beta_2$ 足够大且 $\beta_1<\sqrt{\beta_2}$ 时，常数学习率 Adam 收敛到临界点的邻域，邻域大小正比于梯度噪声方差；**在强增长条件下收敛到临界点本身**。$\beta_2$ 小时存在发散区，收敛与发散之间有相变。Reddi 等的反例[^reddi18]依赖小 $\beta_2$，与"超参选错而非算法缺陷"相容。

### 1.3 Adam 的三种解读：sign descent、对角经验 Fisher、重尾类不平衡

- **sign descent**：Balles–Hennig[^balles18] 把 Adam 更新拆成 $\mathrm{sign}(\hat m)\cdot(1+\mathrm{relvar})^{-1/2}$；Kunstner 等 2023[^kunstner23] 实证大 batch 下 Adam 行为 $\approx$ 带动量的 sign descent，噪声不是 SGD 与 Adam 在 transformer 上差距的主因。
- **对角经验 Fisher**：FAdam[^fadam] 把 $\hat v$ 读成对角经验 Fisher 的在线估计。Kunstner–Hennig–Balles 2019[^kunstner19] 的警告：经验 Fisher $\bar F=\mathbb E[gg^\top]$（真标签）与 Fisher $F$（模型采样标签）的差别在于**模型越拟合，$\bar F$ 越小**——EF 在最优点附近系统性低估曲率。Martens §11 的最简反例：$h=\theta^2/2$ 下 $\bar F=\theta^2\to0$ 而 $F=1$[^martens20]。
- **重尾类不平衡**：Kunstner 等 2024[^kunstner24] 证明在 Zipf 频率的 softmax 问题上，GD 在第 $k$ 类的损失按 $\Theta(1/(\pi_kt))$ 代数衰减、速率 $\propto$ 频率，而 sign descent 按 $\Theta(e^{-ct})$ 指数衰减、频率无关；机制是梯度幅值与 Hessian 幅值跨坐标经 $\pi_k$ 自动相关，所以逐坐标归一化恰好抵消了频率。

### 1.4 噪声下的最优预条件：全白化放大噪声

Martens JMLR 2020 §14 定理[^martens20]：二次目标、迭代 $\theta^+=\theta-\alpha B^{-1}g$、梯度噪声协方差 $\Sigma_g$，常数步长的驻态超额损失为

$$
\mathbb E[h]-h^*\;\to\;\frac{\alpha}{4}\,\mathrm{tr}\bigl((B-\tfrac{\alpha}{2}H)^{-1}\Sigma_g\bigr).
$$

若 $B=H$ 且 realizable（$\Sigma_g=H$），噪声项是 $\alpha n/4$，**正比于参数维度**；NGD 相对一阶法在噪声项上有优势，要求 $H$ 病态**且** $\Sigma_g$ 与 $H$ 对齐。Polyak 平均下噪声主项与 $B$ 无关，预条件的好处是 pre-asymptotic 的。Ollivier 把在线 NGD 写成 Kalman 滤波[^ollivier18]，逐方向增益是阻尼型 $s_i^2/(s_i^2+\sigma_i^2)$ 而非纯逆——低 SNR 的慢 mode 应**降权而不是放大**。

### 1.5 矩阵白化优化器：Shampoo、SOAP、Muon 与 Fisher 的关系

Shampoo[^shampoo] 用 $L^{-1/4}GR^{-1/4}$；Morwani 等[^morwani]证明 Shampoo 所用近似的平方恰是 Hessian 最优 Kronecker 近似的一步幂迭代，即 Shampoo $\approx$（Kronecker 近似 GN）$^{1/2}$，是"开方后的 GN"。SOAP 在 Shampoo 特征基下跑 Adam[^soap]。Muon 的方向 $UV^\top=(GG^\top)^{-1/4}G(G^\top G)^{-1/4}$，即**单样本 Kronecker 因子上的 Shampoo**，也是谱范数下的最陡下降[^bernstein24]；规模化证据见 Liu 等[^muon-scale]。所以 Muon 是把奇异值谱拍平到全 1 的**半白化**（$-1/2$ 次幂而非 $-1$），介于欧氏步与 NGD 之间，与 Adam 同属"符号化"家族。

### 1.6 蒸馏的优化与统计理论

Mobahi 等[^mobahi20]的自蒸馏：每轮对上一轮做岭回归，逐特征方向收缩因子 $d_j/(c+d_j)$，小特征值方向先消失——谱序现象的严格原型（注意他们是迭代自蒸馏，我们是单次师生吸收，机制不同、谱序相同）。Stanton 等[^stanton21]：学生常不能匹配教师，主因是优化不足而非容量——直接支持"吸收是优化受限"的定位。Harutyunyan 等[^harutyunyan23]的 supervision complexity：教师软标签复杂度超出学生容量时蒸馏变差——为跨族平线的 $d_{gap}$ 提供现成词汇。Menon 等[^menon]：软标签相对 one-hot 的方差缩减。

### 1.7 Policy mirror descent 的步长方向

Lan[^lan-spmd]、Xiao[^xiao22]、Johnson 等[^johnson23]：PMD 的线性率来自向策略迭代（$\eta\to\infty$）靠拢，正确方向是 $\eta_k$ **几何递增**，且 Johnson 证明自适应（增大的）步长是达到最优率 $\gamma^k$ 的必要条件。Homotopic PMD[^homotopic]是递减正则 + 递增步长。而随机噪声地板 $\propto\eta\sigma^2$ 时文献支持递减[^lan-spmd]。两方向的分工："精确内解 → 增，噪声地板 → 减"。这在 §3.7 用来裁决 MDPO 的 $\eta$ 调度。

## 2. OPD 文献地图（2025–2026，94 篇核对）

只列与优化/几何直接相关的部分，全部 arXiv id 已抓 abs 页核实。

| 主题                 | 代表工作                                                                        | 与本项目关系                                                    |
| -------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| NGD 自蒸馏           | GeoSD[^geosd]                                                                   | 已占"自然梯度蒸馏"，但是 privileged 自蒸馏 + K-FAC 参数空间     |
| Fisher 投影目标      | FP-OPD[^fpopd]                                                                  | Fisher 用于目标投影，不是预条件器                               |
| 权重更新谱           | 2606.13657[^sparse-opd]、2606.07082[^geom-opd]                                  | 权重空间奇异谱与子空间锁定，**不是 Fisher 谱**                  |
| 近端 / 信赖域        | TOP-D[^topd]、TrOPD[^tropd]、SPOT[^spot]                                        | 策略空间近端 OPD 已有收敛证明                                   |
| sign 型 advantage    | TV-OPD[^tvopd]、SG-OPD[^sgopd]                                                  | **只保留 advantage 符号即可匹配标准 OPD**——与 sign-descent 一致 |
| 方差缩减             | vOPD[^vopd]、PowerOPD[^poweropd]、2603.25562[^revisit-opd]、TIDE、TA-OPD        | 估计量层面已饱和                                                |
| 梯度结构             | 2608.25643[^tokenlevel]、Rock tokens[^rock]、IW-OPD[^iwopd]                     | 逐 token 梯度范数分解；持续高损 token 占梯度大头                |
| 机制 / 天花板        | OPSA[^opsa]、IDA-OPD[^ida]、2608.11829[^tts]、Rethinking I/II[^opd1][^rethink2] | "算法饥饿"、抑制低概率 token、pass@k 不涨                       |
| 优化器 vs 谱（RLVR） | ISO[^iso]                                                                       | AdamW vs Muon 在固定谱上比较，**RLVR 非 OPD**                   |
| 等价框架 / 综述      | G-OPD[^gopd]、两篇综述[^survey1][^survey2]                                      | OPD = 等权 KL 约束 RL                                           |

检索为空的查询：`OPD + mirror descent`、`OPD + natural gradient`（教师 - 学生）、`OPD + preconditioned/Shampoo/Muon`、`OPD + Adam`（仅同名作者）、`OPD + learning rate`（仅 Open-MOPD 的非均匀 lr 诊断[^openmopd]）、`OPD + floor`。**因此以下三件事目前无人写过**：OPD 中优化器选择的理论与实验；OPD 的 Fisher 谱与 $d(t)$ 的拉普拉斯关系；OPD 地板的成因分解与学习率理论。

## 3. 新推导与性质

### 3.1 性质 A：OPD 是结构性插值问题，SGC 成立，乘性噪声

采样 token 估计量的逐样本梯度是 $a_i\,\nabla\log p(y_i\mid s_i)$。若 $p=q$ 在学生访问的每个状态上成立，则每个采样 token 都有 $a_i=0$，**逐样本梯度恒零**——这是插值条件的定义。反过来，噪声协方差

$$
\Sigma_g=\frac1M\,\mathbb E\bigl[a^2\,ss^\top\bigr],\qquad a=\log q-\log p,
$$

在 $p\to q$ 时 $a=O(\Vert\delta z\Vert)$，故 $\Sigma_g=O(d)$：噪声是**乘性**的，随残余距离一起消失。这正是动力学文档 §7.2 噪声分解 $C=\gamma d\cdot H+\nu_0C_0$ 里的第一项，$\gamma\approx3$ 由 gnorm 校准。

推论：把 Vaswani 与 Zhang 的定理搬过来——在 on-policy 状态分布上、若教师在学生族内可达（$d_{gap}=0$），常数学习率的 SGD 与大 $\beta_2$ 的 Adam 在局部二次模型下**都应线性收敛到 $d=0$**，没有经典的 $\alpha\sigma^2$ 型地板。观测到的 $\sim10^{-3}$ 地板于是只能来自：(i) $d_{gap}>0$（跨族臂 0.14 平线是极端例子）；(ii) 加性分量 $\nu_0C_0$（状态分布随策略漂移造成的"移动目标"、context 噪声）；(iii) 数值精度失配（§3.6）。**地板位置因此成为"插值是否成立"的诊断读数**，而不是训练充分度的读数。

冻结臂给出直接证据：`ceejzu26`（状态与锚均冻结、noratio）晚期速率稳定在约 0.009/步且**无地板**（$d$ 从 0.0019 降到 0.0012 后进入估计量噪声），梯度范数降到 0.012；Rayleigh 商代理 $\Vert g\Vert^2/d$ 下降约 240 倍，说明残余集中在低曲率 mode 而非噪声。目标一旦固定，加性分量消失、纯乘性噪声下地板归零——与性质 A 的预言一致。

### 3.2 性质 B：乘性噪声下的地板归零与维度阈值

把 Martens 定理与乘性噪声自洽地放在一起。设 $\Sigma_g(d)=\gamma d\,H/n_{\rm eff}$ 形式（$n_{\rm eff}$ 为噪声有效散布的维度数），预条件 $B$，则驻态满足

$$
d_\infty=\frac{\alpha}{4}\,\mathrm{tr}\bigl(B^{-1}\Sigma_g(d_\infty)\bigr)=\frac{\alpha\gamma}{4n_{\rm eff}}\,\mathrm{tr}(B^{-1}H)\,d_\infty .
$$

这是 $d_\infty$ 的齐次方程：要么 $d_\infty=0$，要么系数为 1。于是

- **SGD**（$B=I$）：$\frac{\alpha\gamma}{4n_{\rm eff}}\mathrm{tr}H<1$ 时地板严格为零，否则发散——乘性噪声不给地板，只给一个稳定性阈值；
- **全白化**（$B=H$）：$\mathrm{tr}(B^{-1}H)=n$，阈值变成 $\alpha\gamma\,n/(4n_{\rm eff})<1$。当噪声散布维度 $n_{\rm eff}$ 远小于参数维度 $n$（OPD 里噪声集中在采样 token 的 score 方向上），**全白化把稳定性阈值收紧 $n/n_{\rm eff}$ 倍**，这是 Martens "realizable 下噪声项 $\propto n$" 的乘性版本；
- **Adam 对角**（$B=\mathrm{diag}\sqrt{\hat v}$）介于两者之间。

性质 B 把"白化上限 2–3 倍且碰不到地板"补成一个更强的陈述：白化在乘性噪声下不改变地板（都是零），但**缩小可用学习率的稳定窗口**；真正抬高地板的是加性分量，而加性分量在白化后按 $\mathrm{tr}(B^{-1}C_0)$ 着色——低曲率方向的加性噪声被放大 $1/\lambda_i$ 倍。这就是 Kalman/Wiener 阻尼增益的来源。

### 3.3 性质 C：OPD 的经验 Fisher 消失，Adam 的有效步长在地板附近膨胀

Adam 的 $\hat v_j\approx\bar g_j^2+\sigma_j^2$。在 OPD 中两项都 $\propto a^2$：信号 $\bar g\propto d$（一阶）、噪声 $\sigma_j^2\propto\gamma d$。于是 $\sqrt{\hat v_j}\propto\sqrt d$，有效步长 $\alpha/\sqrt{\hat v_j}$ 随 $d\to0$ 按 $d^{-1/2}$ **膨胀**。这与 Kunstner 2019 对 EF 的批评是同一件事：模型越拟合教师，EF 越低估曲率，因为真 Fisher $F=\mathbb E_{y\sim p}[ss^\top]$ 不随 $d$ 消失。

两个后果：

1. **信号相是 sign descent、噪声相是白化 GD** 的两相律（动力学文档 §3.2）在 EF 消失下多了一层：进入噪声相后每坐标的有效增益 $\bar g_j/\sqrt{\bar g_j^2+\sigma_j^2}\to\bar g_j/\sigma_j$ 与 $d$ 无关（乘性噪声下 SNR 时间不变），Adam 退化成一个**固定预条件 $P=\mathrm{diag}(1/\sigma_j)$ 的 GD**，其中 $\sigma_j$ 是"逐坐标的噪声标准差除以 $\sqrt d$"——这就是反演谱读到的白化谱 $\tilde\lambda=\rho/2\alpha$ 的来源。
2. **学习率扫描的形状**。若地板由 Adam 驻态决定，应有 $d_\infty\propto\alpha$；实测（全数据 $m=1$ 臂，末 20 步均值）是相反方向：

| $\alpha$           | $d_{\rm last20}$ |
| ------------------ | ---------------- |
| $5{\times}10^{-7}$ | 0.01338          |
| $10^{-6}$          | 0.00550          |
| $2{\times}10^{-6}$ | 0.00267          |
| $5{\times}10^{-6}$ | 0.00129          |
| $10^{-5}$          | 0.00127          |

拟合 $d_\infty\approx A/\alpha+B\alpha$ 得 $A\approx5.0{\times}10^{-9}$、$B\approx79$、$\alpha^*\approx7.9{\times}10^{-6}$、$d_{\min}\approx1.26{\times}10^{-3}$（对数 rmse 0.138）。$1/\alpha$ 项是**有限时间瞬态**（279 步没跑完），$\alpha$ 项是 Adam 型驻态；两项交叉给出一个约 $1.2{\times}10^{-3}$ 的最低值，与 one-shot 臂的 $d_{last}$（0.00075–0.0015）同量级。**这个下限不是 Adam 噪声，是别的东西**（§3.6）。

匹配 $\alpha t$ 的塌缩检验支持同一图像：各 lr 臂在 $\alpha t\le0.5{\times}10^{-4}$ 前重合，之后大 lr 臂每单位 $\alpha t$ 进展更少——sign 型更新的行程不随 $\alpha$ 线性放大。

### 3.4 性质 D：$\beta_2$ 记忆是隐式学习率衰减，Adam 时间是正确的横轴

定义 Adam 时间 $u_t=\sum_{\tau\le t}\alpha\,\overline{|\hat m|/\sqrt{\hat v}}_\tau$，即实际走过的坐标平均步长之和。由于 $\hat v$ 以 $1/(1-\beta_2)\approx10^3$ 步的窗口记忆早期大梯度，而 OPD 梯度按性质 C 随 $d$ 收缩，$\hat v$ 落后于 $g^2$，比值 $|g|/\sqrt{\hat v}$ 从 1 附近**掉到 0.17–0.2**（$\beta_2=0.999$ 臂晚期）。这等价于一个约 5 倍的隐式学习率衰减。$\beta_2=0.95$ 臂（窗口 20 步）该比值保持在 0.8–1。

实测（同 lr $10^{-6}$、同数据）：两臂在 $t\approx140$ 前完全重合，之后 $\beta_2=0.95$ 臂的 $d$ 低 1.4–1.5 倍（末 20 步 0.00378 vs 0.00550）；把两臂画在 $u$ 轴上，曲线重合。**因此 $\beta_2$ 的差别不是几何、是有效学习率调度**。这解释了为何常数 lr 的 Adam 曲线看起来像在"退火"，也说明 "lr 加大只把地板压浅一点"的一部分是 $\hat v$ 记忆的产物。

对照 Zhang 2022 定理：$\beta_2$ 大有利于收敛保证，但代价是这个隐式衰减；$\beta_2=0.95$ 在 OPD 上没有发散，说明当前设置离发散区尚远（$\beta_1=0.9<\sqrt{0.95}$ 满足）。

### 3.5 性质 E：Adam 均衡谱的坐标对齐条件

Kunstner 2024 的机制要求 Fisher 的"慢方向"是**坐标对齐**的（每个类一个权重块）。用 toy 二次模型检验：若 $H$ 在坐标基下对角，Adam 的逐坐标归一化把所有 mode 拉到同一速率，谱色散完全消失；把特征基旋转 45°，$\beta_2=0.999$ 的 Adam **恢复了谱慢化**（慢 mode 又按其特征值衰减），$\beta_2=0.95$ 因为 $\hat v$ 能跟随局部方向而部分恢复均衡。动力学文档 §3.1 已指出 $P$ 与 $F$ 特征基强不对易（对易近似误差达 85 倍）。

推论：OPD 的谱色散里，**token 频率造成的部分**（输出层 unembedding 行、词表方向）是坐标对齐的，Adam 能均衡；**参数共享造成的部分**（中间层的方向、跨 token 耦合）不是，Adam 均衡不了。反演谱里剩下的 25 倍色散应主要是后者。这给了 Muon 式矩阵白化一个明确的可服务对象：它均衡的是每个权重矩阵内部的奇异方向，正好是 Adam 碰不到的那部分。

用反演谱做前向模拟（到 $d/d_{30}=0.01$ 的步数）：

| 谱变换                                                     | 步数 | 相对实际 |
| ---------------------------------------------------------- | ---- | -------- |
| 实际谱                                                     | 618  | 1        |
| 全均衡到均值速率（理想 NGD）                               | 169  | 3.7×     |
| 开方均衡 $\rho_i\to\sqrt{\rho_i\bar\rho}$（Muon 式半白化） | 253  | 2.4×     |
| 全均衡到最大速率（不可达上界）                             | 47   | 13×      |

进度报告的 1.9–2.6 倍是到 0.2/0.1 的数字，越深越大。半白化拿到全白化收益的约三分之二，同时按性质 B 只把稳定阈值收紧 $\sqrt{n/n_{\rm eff}}$ 倍。

### 3.6 性质 F：地板的硬分量与精度失配

性质 A–C 把地板压到一个既非 Adam 噪声也非瞬态的残量 $\approx10^{-3}$（lr 扫描交叉值 $1.26{\times}10^{-3}$；lr $10^{-5}$ 臂晚期对数斜率仅 $-0.13$，接近平）。候选来源：

1. **支撑 gap**（动力学文档 §7.3）：未采样 token 300 步内 83% 概率从未被训。
2. **状态分布漂移**：on-policy 下每步目标移动，$\nu_0C_0$ 项。
3. **rollout 与训练侧的 bf16 失配**：Qi 等[^fp16]指出 bf16 舍入使推理引擎与训练引擎给出的 $\log p$ 系统性不一致，是 RL 微调不稳定的主因，改 fp16 即消除。本项目 rollout（vLLM bf16）与训练（bf16 计算、fp32 主权重）分离，advantage 里的 $\log p$ 来自训练侧重算而 token 是 rollout 侧采的，因而 $d$ 的读数含一个不可被优化消除的 $\mathrm{KL}(\pi_{\rm rollout}\Vert\pi_{\rm train})$ 分量。他们报告的失配量级恰在 $10^{-3}$ nats/token 附近。

现有 run 未开启 `rollout_correction`，日志里没有 `rollout_corr/*` 指标，无法直接读出第 3 项。这是最便宜的判决实验（§5 D1）。

### 3.7 性质 G：MDPO 的 $\eta$ 方向——递减到平台，锚宜 lazy

PMD 理论（§1.7）说精确内解下 $\eta$ 应递增；OPD 的精确 KL 收缩律（动力学文档 §6.2）说 $s=1$ 恒最优；而噪声地板 $\propto\eta\sigma^2$ 支持递减。三者在 OPD 上的统一读法：

- 精确内解 regime 不存在（当前 $\bar\psi\approx0.05$，内层交付率是绑定约束），$\eta$ 与收缩解耦，只经弹簧刚度 $1/\eta$ 起作用（leash 与防腐两通道）——所以文献的"递增"不直接适用；
- 但把 $\eta$ 收到 0.01 让 leash 把所有 mode 绑死、出现尾部反弹，违反 PMD 方向的精神（锚权不该无限增长）；
- 折中就是动力学文档 §6.5 的常数平台 $\eta^*\approx0.2$–0.4。

再加一条从 §1.1 来的方向：当前实现是 greedy 锚（每 outer step 锚重置到 rollout 策略）。lazy 锚（锚滞后 $K$ 步，GeoSD 用 $K=64$）在非平稳目标上累积效应不同：它让 $m$ 步内层看到同一个近端中心，等价于把 prox 的 Bregman 中心固定、由 dual averaging 语义驱动，PMD 里这正是能用递增 $\eta$ 的那一族。这是 E5 消融应加的一维。

## 4. OPD 的本质：一句话与三个特征

**OPD 是在学生自己诱导的状态分布上、对一个固定教师做反向 KL 的指数梯度 / 输出空间自然梯度；参数化把它变成一个插值型、乘性噪声、谱色散来自 token 频率与参数共享两个源的随机优化问题。** 三个特征各对应一个杠杆：

| 特征            | 含义                                                    | 杠杆                                                        |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| 插值 + 乘性噪声 | 理论上无经典地板；地板是 gap / 漂移 / 精度的诊断量      | 修读数（精度、支撑）比调优化器更先                          |
| 双源谱色散      | 频率源坐标对齐、共享源不对齐                            | Adam 管前者，矩阵半白化管后者，输出空间自然化管两者的目标侧 |
| EF 消失         | Adam 的 $\hat v$ 塌陷、有效步长膨胀、$\beta_2$ 隐式退火 | $\hat v$ 下限 / 真 Fisher 对角 / Adam 时间调度              |

TV-OPD 的"只留 advantage 符号即可"[^tvopd]与 OPSA 的"固定负 advantage 即可"[^opsa]在这一图像下都不奇怪：sign descent 本来就只用方向，advantage 的幅值在 Adam 里几乎被 $\hat v$ 归一化掉了。这也提示一个反向检验：在 SGD 上 sign(a) 应显著差于 $a$。

## 5. 改进 Adam+OPD：提案与诊断

按"先修读数、再谈干预"的项目规则排序。

**D. 零成本诊断**

- D1 **精度失配**：开启 `rollout_correction` 的指标输出（不施加 IS 权重），读 $\mathrm{KL}(\pi_{\rm rollout}\Vert\pi_{\rm train})$ 逐步曲线；若其量级与 $d$ 地板相当，则地板的硬分量是 bf16，处方是 fp16 或对齐 kernel，而非任何优化器改动。
- D2 **Adam 时间对齐**：把现有 lr 扫描与 $\beta_2$ 臂全部画到 $u$ 轴上；若全部重合，则"lr / $\beta_2$ 对地板的影响"整体归结为有效步长调度，后续只需调度、不需换优化器。
- D3 **噪声 - 曲率对齐比**：在冻结 checkpoint 上估 $\mathrm{tr}(F^{-1}\Sigma_g)/\mathrm{tr}(\Sigma_g)\cdot\bar\lambda$（Lanczos 求 $F$ 的前若干特征对，$\Sigma_g$ 用逐样本梯度子采样）。比值远小于 1 说明噪声躲在高曲率方向、白化安全；接近或大于 1 说明性质 B 的阈值收紧会咬人。
- D4 **sign 充分性**：SGD 上跑 sign(a) vs $a$；Adam 上二者应几乎相同。

**O. 优化器层干预**

- O1 **$\hat v$ 下限或 $\beta_2$ 降低**（最便宜）：$\beta_2=0.95$ 已给 1.4–1.5 倍；另一形式是 $\hat v\leftarrow\max(\hat v,\,v_{\rm floor})$ 或用真 Fisher 对角替代 EF——采样 $y'\sim p$ 做一次额外反传得到 score 的逐样本平方需要逐样本梯度，成本高，可在小子集上估、按层广播。
- O2 **矩阵参数上的 Muon 式半白化**：ISO 在 RLVR 上已展示 AdamW 与 Muon 在固定谱框架下可换[^iso]。预测：到 $d/d_{30}=0.01$ 的步数从 618 降到约 253（2.4 倍）；若 D3 的对齐比大，收益会被噪声抵消。这一项在 OPD 上无人做过。
- O3 **Adam 时间学习率调度**：以 $u$ 而非 $t$ 作调度变量，抵消 $\hat v$ 的隐式退火（等价于对 $\alpha$ 乘 $\sqrt{\hat v}/\sigma$ 的估计）。

**T. 目标层干预**

- T1 **输出空间自然化**：CLR（`2026-09-09-centered-logit-regression-fisher-opd.md`）或 FP-OPD 式 Fisher 投影，把 logits 空间的 NGD 免费拿到手，再交给 Adam；由 §1.1，这是精确的输出空间 MD，与参数空间预条件正交、可叠加。上限仍是谱均衡的 2–4 倍。
- T2 **MDPO 调度**：常数 $\eta\in[0.2,0.4]$ + lazy 锚 $K\in\{1,64\}$（§3.7）。

**明确不做**：D1–D3 出结果前不写 K-FAC；不用跨族对测优化器差异。

## 6. 开放问题

1. 性质 B 的 $n_{\rm eff}$：OPD 噪声的有效散布维度到底多大（决定白化的安全边际），需 D3 测。
2. 性质 E 的双源分解比例：频率源与共享源各占反演谱色散的多少，可用 unembedding 层单独冻结 / 单独训练来分离。
3. 精度失配是否就是 $10^{-3}$ 硬地板：D1 一步判决。
4. lazy 锚在 OPD 上是否恢复 PMD 的递增 $\eta$ 方向。

### 参考文献

[^raskutti]: G. Raskutti, S. Mukherjee. The Information Geometry of Mirror Descent. IEEE Trans. Inf. Theory 2015. [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)

[^amari98]: S. Amari. Natural Gradient Works Efficiently in Learning. Neural Computation 10(2), 1998.

[^mcmahan]: H. B. McMahan. A Survey of Algorithms and Analysis for Adaptive Online Learning. JMLR 2017. [arXiv:1403.3465](https://arxiv.org/abs/1403.3465)

[^schmidt13]: M. Schmidt, N. Le Roux. Fast Convergence of Stochastic Gradient Descent under a Strong Growth Condition. [arXiv:1308.6370](https://arxiv.org/abs/1308.6370)

[^vaswani19]: S. Vaswani, F. Bach, M. Schmidt. Fast and Faster Convergence of SGD for Over-Parameterized Models and an Accelerated Perceptron. AISTATS 2019. [arXiv:1810.07288](https://arxiv.org/abs/1810.07288)

[^ma18]: S. Ma, R. Bassily, M. Belkin. The Power of Interpolation: Understanding the Effectiveness of SGD in Modern Over-parametrized Learning. ICML 2018. [arXiv:1712.06559](https://arxiv.org/abs/1712.06559)

[^zhang22adam]: Y. Zhang, C. Chen, N. Shi, R. Sun, Z.-Q. Luo. Adam Can Converge Without Any Modification On Update Rules. NeurIPS 2022. [arXiv:2208.09632](https://arxiv.org/abs/2208.09632)

[^reddi18]: S. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. ICLR 2018. [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)

[^balles18]: L. Balles, P. Hennig. Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients. ICML 2018. [arXiv:1705.07774](https://arxiv.org/abs/1705.07774)

[^kunstner23]: F. Kunstner, J. Chen, J. W. Lavington, M. Schmidt. Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be. ICLR 2023. [arXiv:2304.13960](https://arxiv.org/abs/2304.13960)

[^fadam]: D. Hwang. FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information. [arXiv:2405.12807](https://arxiv.org/abs/2405.12807)

[^kunstner19]: F. Kunstner, L. Balles, P. Hennig. Limitations of the Empirical Fisher Approximation for Natural Gradient Descent. NeurIPS 2019. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)

[^martens20]: J. Martens. New Insights and Perspectives on the Natural Gradient Method. JMLR 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^kunstner24]: F. Kunstner, R. Yadav, A. Milligan, M. Schmidt, A. Bietti. Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models. NeurIPS 2024. [arXiv:2402.19449](https://arxiv.org/abs/2402.19449)

[^ollivier18]: Y. Ollivier. Online Natural Gradient as a Kalman Filter. Entropy 2018. [arXiv:1703.00209](https://arxiv.org/abs/1703.00209)

[^shampoo]: V. Gupta, T. Koren, Y. Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568)

[^morwani]: D. Morwani, I. Shapira, N. Vyas, E. Malach, S. Kakade, L. Janson. A New Perspective on Shampoo's Preconditioner. [arXiv:2406.17748](https://arxiv.org/abs/2406.17748)

[^soap]: N. Vyas et al. SOAP: Improving and Stabilizing Shampoo using Adam. ICLR 2025. [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)

[^bernstein24]: J. Bernstein, L. Newhouse. Old Optimizer, New Norm: An Anthology. [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)

[^muon-scale]: J. Liu et al. Muon is Scalable for LLM Training. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

[^mobahi20]: H. Mobahi, M. Farajtabar, P. Bartlett. Self-Distillation Amplifies Regularization in Hilbert Space. NeurIPS 2020. [arXiv:2002.05715](https://arxiv.org/abs/2002.05715)

[^stanton21]: S. Stanton, P. Izmailov, P. Kirichenko, A. Alemi, A. G. Wilson. Does Knowledge Distillation Really Work? NeurIPS 2021. [arXiv:2106.05945](https://arxiv.org/abs/2106.05945)

[^harutyunyan23]: H. Harutyunyan, A. S. Rawat, A. K. Menon, S. Kim, S. Kumar. Supervision Complexity and its Role in Knowledge Distillation. ICLR 2023. [arXiv:2301.12245](https://arxiv.org/abs/2301.12245)

[^menon]: A. K. Menon, A. S. Rawat, S. Reddi, S. Kim, S. Kumar. Why distillation helps: a statistical perspective. ICML 2021. [arXiv:2005.10419](https://arxiv.org/abs/2005.10419)

[^lan-spmd]: G. Lan. Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes. Math. Programming 2022. [arXiv:2102.00135](https://arxiv.org/abs/2102.00135)

[^xiao22]: L. Xiao. On the Convergence Rates of Policy Gradient Methods. JMLR 2022. [arXiv:2201.07443](https://arxiv.org/abs/2201.07443)

[^johnson23]: E. Johnson, C. Pike-Burke, P. Rebeschini. Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes. NeurIPS 2023. [arXiv:2302.11381](https://arxiv.org/abs/2302.11381)

[^homotopic]: Y. Li, G. Lan, T. Zhao. Homotopic Policy Mirror Descent. Math. Programming 2023. [arXiv:2201.09457](https://arxiv.org/abs/2201.09457)

[^geosd]: I. Jukić, I. Titov. Geometric Self-Distillation for Reasoning Generalization. [arXiv:2607.06855](https://arxiv.org/abs/2607.06855)

[^fpopd]: Xue et al. Distill What the Student Can See: Fisher-Projected On-Policy Distillation. [arXiv:2608.01263](https://arxiv.org/abs/2608.01263)

[^sparse-opd]: Yu et al. Dense Supervision, Sparse Updates: Sparsity and Geometry of On-Policy Distillation. [arXiv:2606.13657](https://arxiv.org/abs/2606.13657)

[^geom-opd]: Shen et al. On the Geometry of On-Policy Distillation. [arXiv:2606.07082](https://arxiv.org/abs/2606.07082)

[^topd]: Xie et al. Trust Region Policy Distillation. [arXiv:2607.04751](https://arxiv.org/abs/2607.04751)

[^tropd]: Xing et al. Trust Region On-Policy Distillation. [arXiv:2606.01249](https://arxiv.org/abs/2606.01249)

[^spot]: Lin, Han. SPOT: Surgical Post-Training. [arXiv:2603.01683](https://arxiv.org/abs/2603.01683)

[^tvopd]: Xiao et al. TV-Regulated On-Policy Distillation. [arXiv:2609.08341](https://arxiv.org/abs/2609.08341)

[^sgopd]: Xu et al. SG-OPD: Sign-Gated On-Policy Distillation. [arXiv:2606.09304](https://arxiv.org/abs/2606.09304)

[^vopd]: Oh et al. KL for a KL: Value-Baselined On-Policy Distillation. [arXiv:2605.07865](https://arxiv.org/abs/2605.07865)

[^poweropd]: Zhao et al. PowerOPD. [arXiv:2606.17199](https://arxiv.org/abs/2606.17199)

[^revisit-opd]: Fu et al. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. [arXiv:2603.25562](https://arxiv.org/abs/2603.25562)

[^tokenlevel]: Shao et al. Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation. [arXiv:2608.25643](https://arxiv.org/abs/2608.25643)

[^rock]: Jiang et al. Rock Tokens. [arXiv:2605.09253](https://arxiv.org/abs/2605.09253)

[^iwopd]: Xie et al. IW-OPD: Importance-Weighted On-Policy Distillation. [arXiv:2606.22600](https://arxiv.org/abs/2606.22600)

[^opsa]: Ding, Zhang. Does On-Policy Distillation Really Distill? [arXiv:2608.31046](https://arxiv.org/abs/2608.31046)

[^ida]: Yang et al. IDA-OPD. [arXiv:2608.29846](https://arxiv.org/abs/2608.29846)

[^tts]: Ge et al. Understanding On-Policy Distillation via Test-Time Scaling. [arXiv:2608.11829](https://arxiv.org/abs/2608.11829)

[^opd1]: Li et al. Rethinking On-Policy Distillation of Large Language Models I. [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)

[^rethink2]: Fu et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^iso]: Zhu et al. ISO: An RLVR-Native Optimization Stack. [arXiv:2607.19331](https://arxiv.org/abs/2607.19331)

[^gopd]: Yang et al. G-OPD / ExOPD: On-Policy Distillation as Dense KL-Constrained RL. [arXiv:2602.12125](https://arxiv.org/abs/2602.12125)

[^survey1]: B. Zhang. A Formula-Driven Survey and Research Agenda for On-Policy Distillation. [arXiv:2606.22793](https://arxiv.org/abs/2606.22793)

[^survey2]: Song, Zheng. A Survey of On-Policy Distillation for LLMs. [arXiv:2604.00626](https://arxiv.org/abs/2604.00626)

[^openmopd]: Gao et al. Open-MOPD. [arXiv:2608.19098](https://arxiv.org/abs/2608.19098)

[^fp16]: P. Qi, Z. Liu, X. Zhou, T. Pang, C. Du, W. S. Lee, M. Lin. Defeating the Training-Inference Mismatch via FP16. [arXiv:2510.26788](https://arxiv.org/abs/2510.26788)
