# 2026-09-06: OPD 吸收率下降的谱吸收模型（阅读笔记、推导与实验方案）

对象论文：Fu et al. (2026)[^fu26]，_Rethinking On-Policy Distillation of Large Language Models II: One Training Example_（下称"论文"）。论文的动机来自 one-shot RLVR（Wang et al. 2026[^wang26a]）；姊妹篇 Part I 处理数据侧的现象学与机制[^opd1-26]；工程配方与术语出自 Thinking Machines 的博文（Lu 2025[^tml25]）；综述见 Song & Zheng (2026)[^song26]。本文回答论文留下的 open question："What sets the absorption rate is still open."

**核心论点一句话**：吸收率 $v_t$ 下降不需要新机制来解释——任何一阶优化器在宽谱（幂律谱）曲率上的行为必然如此。在解附近，OPD 动力学按 Hessian/Fisher 特征方向解耦成独立的指数弛豫模式；每一步吃掉的剩余距离比例 $\approx \eta \times \langle\lambda\rangle(t)$，即学习率乘以**剩余误差质量的平均曲率**。高曲率模式先被耗尽，剩余质量不断迁入平坦方向，$\langle\lambda\rangle(t)$ 单调下降，于是 $v_t$ 单调下降。衰减指数是"架构 × 目标几何"的性质，与数据量无关——这正是论文观察到的数据无关性的原因，而且论文自己的两个控制实验（学习率折叠、冻结状态）恰好是这个模型的两个可检验预言[^fu26]。

---

## 1. 论文事实清单

以下事实均转写自论文[^fu26]。

### 1.1 目标与度量

OPD 目标（论文 Eq. 1）：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_{\theta}}\left[\sum_{i=1}^{L}\operatorname{KL}\!\left(\pi_{\theta}(\cdot\mid s_{i})\,\|\,\pi_{T}(\cdot\mid s_{i})\right)\right],
$$

即学生采样状态、在**每个访问状态上最小化 reverse KL**；实践估计器是逐 token advantage（Eq. 2）：$A_i = \log\pi_T(y_i\mid s) - \log\pi_θ(y_i\mid s)$，另有 top-$k$ 截断版本（Eq. 3）。这一"学生采样 + reverse KL + 策略梯度估计"的组合源自 MiniLLM（Gu et al. 2024[^gu24]）与 GKD（Agarwal et al. 2024[^agarwal24]）；在"状态由学生自己产生"的意义上，它与 DAgger 的 on-policy 模仿学习同源（Ross et al. 2011[^ross11]）。

两个对齐度量（论文 Eq. 6/7）：

$$
d_t=\frac{1}{|\mathcal{T}_{t}|}\sum_{i\in\mathcal{T}_{t}}\left|\log\pi_{T}(y_{i}\mid s_{i})-\log\pi_{\theta_{t}}(y_{i}\mid s_{i})\right|,\qquad
v_t=\frac{d_t-d_{t+1}}{d_t}.
$$

$v_t$ 用几何窗口 $[t/1.4,\,1.4t]$ 内 $\log d$ 的斜率估计；度量从 step 30 开始（此前 gradient clipping 在生效）。

### 1.2 关键实验数字

配置：学生 R1-Distill-1.5B（另有 Llama-3B / OLMo-7B 对照），AdamW，lr $=10^{-6}$（$\beta_1=0.9$，$\beta_2=0.999$，wd $=0.01$），batch = 64 rollouts，采样温度 1.0，梯度范数 clip 1.0。

| 量               | 1 query | 4    | 16   | 17k（full） |
| ---------------- | ------- | ---- | ---- | ----------- |
| $d_{300}/d_{30}$ | 0.16    | 0.17 | 0.18 | 0.22        |
| $v_{50}/v_{200}$ | 5.6     | 5.2  | 4.2  | 6.4         |

三个结构性的观察：

1. **数据无关**：四个 run 吃掉 step-30 距离的 78%–84%，减速倍数接近；1 题与 17k 题节奏相同（论文 §5.2）。
2. **学习率只重标定时间轴**（论文附录 C.2）：lr 减半/加倍后，把横轴换成 $u=(\mathrm{lr}/10^{-6})\,t$，三条曲线坍缩（离散度 $2.2\times\to 1.1\times$），曲线形状不变——"It bends flatter as it goes"。
3. **冻结状态不缩短 run**（论文 §5.3）：初始学生采 64 条轨迹后全程复用（"always off-policy"），run 依然稳涨约 **200 步**才停，对齐指标节奏不变。

### 1.3 已被实验排除的解释

- **状态供给论**（"run 长是因为新状态源源不断"）：§5.3 的冻结状态实验直接否定——固定状态集本身就够跑几百步。
- **测量漂移论**（"$d_t$ 的采样测度随学生变化，剩余距离被越来越稀有的状态承载"）：同样的实验否定——测度完全冻结后慢节奏原封不动。
- **数据量/梯度冲突论**（"很多题的梯度互相打架，题越多越慢"）：四个数据规模的节奏相同，且 1 题（无冲突）不比 17k 题快。

结论：瓶颈在**参数空间的优化本身**。这就是第 2 节要解释的对象。

---

## 2. 谱吸收模型

### 2.1 推导一：KL 的二次化与 Fisher 度量

冻结状态集 $\mathcal{S}$ 后，目标是一个固定的经验目标 $J(\theta)=\mathbb{E}_{s\in\mathcal{S}}\,[\operatorname{KL}(\pi_\theta(\cdot\mid s)\,\|\,\pi_T(\cdot\mid s))]$。设在 $\theta^{\ast}$ 处 $\pi_{\theta^{\ast}}=\pi_T$（可表示情形；不可表示时下面全部结论只需 $J$ 在 $\theta^{\ast}$ 附近有半正定 Hessian）。取扰动 $\delta=\theta-\theta^{\ast}$，记 $a(y)=\log\pi_{\theta^{\ast}+\delta}(y)-\log\pi_{\theta^{\ast}}(y)=\nabla\log\pi_{\theta^{\ast}}(y)\cdot\delta+O(\delta^2)$。则

$$
\operatorname{KL}(\pi_{\theta^{\ast}+\delta}\,\|\,\pi_{\theta^{\ast}})
= \mathbb{E}_{\theta^{\ast}+\delta}[a]
= \mathbb{E}_{\theta^{\ast}}\big[(1+a+\tfrac12 a^2+\cdots)\,a\big]
= \mathbb{E}_{\theta^{\ast}}[a]+\mathbb{E}_{\theta^{\ast}}[a^2]+O(\delta^3).
$$

其中 $\mathbb{E}_{\theta^{\ast}}[\nabla\log\pi]=0$（score 的零均值），且 $\mathbb{E}_{\theta^{\ast}}[\nabla^2\log\pi]=-F$（Fisher 恒等式），故

$$
\mathbb{E}_{\theta^{\ast}}[a] = \tfrac12\,\delta^\top(-F)\,\delta,\qquad
\mathbb{E}_{\theta^{\ast}}[a^2] = \delta^\top F\,\delta
\quad\Longrightarrow\quad
J(\theta)-J(\theta^{\ast}) \approx \tfrac12\,\delta^\top F\,\delta .
$$

即 **reverse-KL 的局部曲率就是 Fisher 信息度量**（信息几何标准结果，Amari 1998[^amari98]；Amari & Nagaoka 2000[^amari00]）。在 $F$ 的特征基下写 $\delta=\sum_i x_i e_i$、$F e_i=\lambda_i e_i$：

$$
J(\theta)\approx J(\theta^{\ast})+\tfrac12\sum_i \lambda_i x_i^2,\qquad
\text{GD 每步： } x_i \leftarrow (1-\eta\lambda_i)\,x_i .
$$

**各模式完全解耦**：梯度下降在一个二次型上就是一组独立的指数弛豫，速率 $\eta\lambda_i$。这不依赖"网络是线性的"这类强假设——只需要剩余误差小到二阶展开成立（论文从 step 30 开始度量、且 clip 已不再生效，正好落在这个窗口）。NTK/lazy 训练文献给出了同一结论的非线性版本：宽网络训练动力学由核（Fisher/NTK）谱决定（Jacot et al. 2018[^jacot18]、Arora et al. 2019[^arora19]、Chizat et al. 2019[^chizat19]）。

**论文附录 C.2[^fu26] 的学习率折叠是这个线性动力学的签名。** 若 $d$ 由一族 $e^{-\eta\lambda_i t}$ 支配，则 $d$ 只通过组合变量 $\eta t$ 依赖训练——学习率改变的只是沿同一条形状曲线行进的速度。实测坍缩到 $1.1\times$，残余的 $\sim10\%$ 恰好是"on-policy 状态漂移 + Adam 高阶效应"这类二阶量的量级。若慢节奏来自 KL 几何的某种内禀非线性，换 lr 不会是纯粹的时间轴重标定。

### 2.2 推导二：吸收率 = 剩余误差的平均曲率

线性化后每个模式贡献一个独立衰减的分量。把论文的 $d_t$（一阶型距离）写成模式幅度之和

$$
d(t)=\sum_{i\ge1} a_i\,e^{-\eta\lambda_i t},\qquad a_i\ge0,
$$

（严格说 $d$ 是 $|\log\text{-ratio}|$ 的均值而非逐模式幅度之和；在线性化区间内逐模式的符号混合是高阶小量，此假设在 §7 局限里声明。）则

$$
v_t = 1-\frac{d(t+1)}{d(t)}
= \sum_i w_i(t)\,\big(1-e^{-\eta\lambda_i}\big),\qquad
w_i(t)=\frac{a_i e^{-\eta\lambda_i t}}{d(t)} .
$$

对小步长 $1-e^{-\eta\lambda}\approx\eta\lambda$，得到核心等式：

$$
\boxed{\;v_t \;\approx\; \eta\,\langle\lambda\rangle(t),\qquad
\langle\lambda\rangle(t)=\sum_i w_i(t)\,\lambda_i=\text{剩余误差质量的曲率加权平均}.\;}
$$

**解读：论文的吸收率曲线就是剩余误差的"谱质心"读数。** 训练是一个吸收前沿（absorption front）从高曲率模式向低曲率模式的扫掠：快模式先被耗尽，权重 $w_i(t)$ 不断向小 $\lambda$ 倾斜，$\langle\lambda\rangle(t)$ 下降，$v_t$ 随之下降——不需要任何额外机制。

而且这个下降是**单调且可定量**的。对权重分布求时间导数：

$$
\frac{d\langle\lambda\rangle}{dt}
= \sum_i \dot w_i\,\lambda_i
= -\eta\sum_i w_i\lambda_i(\lambda_i-\langle\lambda\rangle)
= -\eta\,\operatorname{Var}_{w(t)}(\lambda)\;\le\;0 ,
$$

等号成立当且仅当剩余质量全部集中在一个 $\lambda$ 上。于是得到一个可以从**现有训练日志**直接算的诊断量：

$$
\operatorname{Var}_{w(t)}(\lambda) \approx -\frac{1}{\eta^2}\,\frac{d v_t}{dt}
\qquad(\text{对 } v_t \text{ 平滑后}) .
$$

**吸收率下降的斜率本身度量了剩余误差的谱方差。** 若某天 $v_t$ 走平，说明剩余误差已经塌缩到单一时间尺度（或噪声底）；若 $v_t$ 陡降，说明前沿正在穿过一段很宽的谱。

### 2.3 与论文数据的三个定量对照[^fu26]

**(a) 幂律衰减的指纹：$v_{50}/v_{200}$ 应恰为 4。** 若 $d(t)\propto t^{-\alpha}$，则 $v_t = 1-(t/(t+1))^\alpha \approx \alpha/t$，与 $\alpha$ 无关地有

$$
v_{50}/v_{200} = \frac{\alpha/50}{\alpha/200} = 4 .
$$

实测（论文 Table 7）四个 run 为 **5.6 / 5.2 / 4.2 / 6.4**——四组独立随机性全部落在 4 附近（轻微偏高与几何窗口估计器 + 底部效应相容）。作为对照：若收敛是干净的单一指数，$v_t$ 为常数，比值应为 1——实测远大于 1。同时由 $d_{300}/d_{30}\approx0.19$ 解出

$$
10^{-\alpha}\approx 0.19 \;\Longrightarrow\; \alpha\approx 0.72 ,
$$

即距离按 $t^{-0.7}$ 幂律下降、吸收率按 $\sim 1/t$ 下降，两个读数自洽。

**(b) 数据无关性是预测而非巧合。** 衰减指数由谱 $\{\lambda_i\}$ 和振幅 $\{a_i\}$ 的谱向分布决定，而谱由学生架构与目标几何决定；数据量只改变哪些方向上有误差质量（振幅）与每步噪声（batch 内多样性），不改变指数。1 题 ≈ 17k 题是必然。

**(c) 学习率折叠残留 1.1×**：线性化预测完美坍缩（1.0×）；残留部分应主要来自 on-policy 状态漂移与 Adam 的高阶效应，且应在**冻结状态的 off-policy run 里进一步缩小**——这是对论文 C.2 的一个可直接做的精化检验。

### 2.4 推导三：幂律谱 ⇒ 幂律 / 拉伸指数尾巴

深网的 Hessian / Fisher / NTK 谱普遍呈跨多个数量级的幂律：经验上，Hessian 特征值密度是幂律 bulk 加少量离群值（Ghorbani et al. 2019[^ghorbani19]、Papyan 2020[^papyan20]）；全连接网络初始化时 NTK 谱按频率多项式衰减（Basri et al. 2020[^basri20]）。设

$$
\lambda_i = \lambda_1\, i^{-\beta},\qquad a_i = a_1\, i^{-\delta}\ (\delta>1\ \text{保证总距离有限}) .
$$

把求和换成积分（本质上是对 Laplace 变换做 Watson 引理渐近）：

$$
d(t)\;\approx\; a_1\int_0^\infty x^{-\delta}\,e^{-\eta\lambda_1 t\,x^{-\beta}}\,dx .
$$

换元 $u=\eta\lambda_1 t\,x^{-\beta}$：

$$
d(t)\;\approx\;\frac{a_1}{\beta}\,(\eta\lambda_1\, t)^{-\frac{\delta-1}{\beta}}\,
\Gamma\!\Big(\frac{\delta-1}{\beta}\Big)
\;=\;C\,t^{-\alpha},\qquad
\boxed{\;\alpha=\frac{\delta-1}{\beta}\;}
$$

一个更一般的表述：把"初始误差质量在曲率谱上的分布"记为测度 $\mu$，$d(t)=\int e^{-\eta\lambda t}\,\mu(d\lambda)$ 是 $\mu$ 的 Laplace 变换。若 $\mu$ 的密度在 $\lambda\to0$ 处满足 $\rho(\lambda)\propto\lambda^{\gamma-1}$，则 $d(t)\sim t^{-\gamma}$；若 $\mu$ 集中在有限带宽上，早期近似指数、尾部由低频端形状决定。这也是"梯度下降作为谱滤波器"理论的时域版本：GD 对第 $i$ 个谱分量的收缩由滤波函数 $(1-(1-\eta\lambda)^t)/\lambda$ 决定，早停等价于谱正则化，偏差消解速率由谱衰减指数决定（Yao, Rosasco & Caponnetto 2007[^yao07]、Caponnetto & De Vito 2007[^caponnetto07]、Dieuleveut & Bach 2016[^dieuleveut16]）。

**无论哪种形态，$v_t$ 都递减**；两者的判别是拟合问题（§5 D1）：

- 振幅沿谱衰减（$\delta>1$）→ **纯幂律** $t^{-\alpha}$；
- 振幅近均匀分布在有限/对数均匀的谱上 → 早期近指数、尾部更快或呈**拉伸指数** $e^{-(t/\tau)^{\gamma}}$（宽谱弛豫系统的标准形态，玻璃动力学里即 Kohlrausch–Williams–Watts 律）。

代入论文数字做一个量级自洽：$\alpha\approx0.72$。若振幅平方约按 $i^{-2}$ 衰减（$\delta\approx2$）、谱指数 $\beta\approx1.4$，则 $\alpha=(\delta-1)/\beta\approx0.71$——$\beta\in[1,2]$ 正是经验 Hessian/NTK 谱的典型区间（Ghorbani et al. 2019[^ghorbani19]）。这只是自洽性展示，不是辨识；真正的辨识交给 D1/D5。

**"几百步而不是几十步"由此得到解释**：高曲率模式贡献了 $d$ 的前段下降（几十步内吃完），而幂律谱的尾部——大量 $\lambda\lesssim1/(\eta t)$ 的模式——承载了之后的长尾。前沿位置 $i^{\ast}(t)\sim(\eta\lambda_1 t)^{1/\beta}$：吃掉的模式数随 $t$ 只按多项式增长。

### 2.5 推导四：噪声底、可吸收性阈值与 run 终止

每个模式上，SGD 是经典的 Ornstein–Uhlenbeck 过程：

$$
x_{t+1}=(1-\eta\lambda_i)\,x_t+\eta\,\xi_t,\qquad
\mathbb{E}[\xi_t^2]=\sigma_i^2/B ,
$$

（$\sigma_i^2$ 为模式 $i$ 的单样本梯度噪声方差，$B$ 为 batch；on-policy 的状态采样也计入 $\sigma_i$。）平稳方差

$$
\mathbb{E}[x_i^2]_\infty
=\frac{\eta^2\,\sigma_i^2/B}{1-(1-\eta\lambda_i)^2}
\;\approx\;\frac{\eta\,\sigma_i^2}{2\lambda_i B}.
$$

（经典 OU 结果，Mandt et al. 2017[^mandt17]。）

两个推论：

1. **平坦模式又慢又吵**：它的均衡位移 $\propto 1/\lambda_i$ 反而最大。
2. **存在可吸收性阈值**：模式 $i$ 能被吸收，要求初始幅度超过自身噪声底：

$$
c_i^2 \gg \frac{\eta\,\sigma_i^2}{2\lambda_i B}
\quad\Longleftrightarrow\quad
\lambda_i \;\gg\; \lambda_{\mathrm{crit}} := \frac{\sigma_i}{c_i}\sqrt{\frac{\eta}{2B}} .
$$

于是 run 的终点由两部分决定：所有 $\lambda>\lambda_{\mathrm{crit}}$ 的模式各自均衡的时间（$t_i\sim\ln(c_i/x_i^{\ast})/(\eta\lambda_i)$ 的最大值），加上谱中 $\lambda<\lambda_{\mathrm{crit}}$ 部分贡献的**不可约底** $d_\infty$。

- **Off-policy run 为什么在 ~200 步停下**：状态冻结 → $\sigma_i$ 固定 → 可吸收模式集合固定 → 全部均衡后曲线进入硬平台。
- **On-policy run 为什么不停**：学生持续采出新状态，把质量重新注入剩余模式（并维持较高的 $\sigma_i$），平台变成 $3$ 分以内的漂移带（论文 Fig. 1 的 1000 步行为[^fu26]）。
- **底的可检验标度**：$d_\infty^2\propto\eta/B$（噪声底部分），且与 $\eta$ 无关的部分是表示缺口（容量/温度/tokenizer 失配）。见 §5 D2。

### 2.6 推导五：信息几何判词——恒定吸收率 = 自然梯度，及其估计论极限

若每次更新用精确自然梯度 $\theta\leftarrow\theta-\eta F^{-1}\nabla J$，在模式坐标里 $x_i\leftarrow(1-\eta)x_i$：**所有模式同速收缩**，$v\equiv\eta$ 恒定，对数距离图是一条直线。所以"$v_t$ 下降"的信息几何等价形式是：**曲率没有被白化，谱就会被按曲率从高到低的顺序消费**。自然梯度（Amari 1998[^amari98]）、自然策略梯度（Kakade 2002[^kakade02]）、信赖域方法（Schulman et al. 2015[^schulman15]、2017[^schulman17]）、镜像下降策略优化（Tomar et al. 2021[^tomar21]）都是这件事的不同近似。

但精确白化在估计论上不可达，而且这个极限不是实现问题：

- 模式 $i$ 的每步梯度信号 $\sim c_i\lambda_i$，噪声 $\sim\sigma_i/\sqrt{B}$；要把该模式白化（用 $g_i/\lambda_i$ 代替 $g_i$），需要信噪比 $\sqrt{B}\,c_i\lambda_i/\sigma_i \gg 1$——**与推导四的 $\lambda_{\mathrm{crit}}$ 是同一个阈值**。在 $\lambda_i\to0$ 的地方，逆预条件子的方差爆炸：**你无法白化你测不准的方向**（Martens 2014[^martens14] 对自然梯度病态性与 damping 的分析）。
- 随机凸优化的信息论下界说明这不是 SGD 特有的缺陷：噪声 oracle 下即便最优算法也只有 $O(1/t)$ 型速率（Agarwal et al. 2012[^agarwal12]、Bach & Moulines 2013[^bach-moulines-13]）；要往下走必须降方差（加大 batch / 方差缩减 / 迭代平均，见 Dieuleveut, Flammarion & Bach 2017[^dieuleveut-17]）。

这给出 "data-overfed but algorithm-starved" 的定量形式：**食物（监督）管够，嘴不够——嘴的大小 = 曲率 × 信噪比。** 谱中 $\lambda<\lambda_{\mathrm{crit}}$ 的那部分监督，无论喂多少数据都吃不到；它们不是"没用的监督"，而是"这个优化器在这个 batch 尺寸下吃不到的监督"。

### 2.7 AdamW 的软底（假设，可证伪）

$\beta_2=0.999$ 意味着二阶矩 EMA 的记忆长度 $\approx 1/(1-\beta_2)=1000$ 步——与 run 的总长度同量级，值得怀疑。一步启发式分析：在噪声主导的平坦模式上，$\mathbb{E}[g^2]\approx\sigma_i^2/B+\lambda_i^2 x_i^2\approx\sigma_i^2/B$，Adam 的有效漂移为

$$
\Delta x_i \approx \eta\,\frac{g_i}{\sqrt{\hat v}}\;\approx\;\eta\,\frac{\lambda_i x_i}{\sigma_i/\sqrt{B}},
$$

即有效收缩率从 $\eta\lambda_i$ 被压到 $\eta\lambda_i\cdot\mathrm{SNR}$；再叠加 $\hat v$ 的长记忆（信号已衰减而分母未跟上），等于在谱尾部又垫了一层"软底"。Adam 末端的此类病态是老话题（Reddi et al. 2018[^reddi18]）。**预测**：$\beta_2\in\{0.95,0.999\}$ 扫描、$\epsilon$ 扫描、末期换优化器，尾巴形态应随之改变（§5 D3）。这一条是假设 H3，不是结论。

---

## 3. 文献对照：这不是一个孤立现象

- **统计学（可靠性理论）**：$v_t$ 本质是"剩余距离"的**风险率（hazard rate）**。经典结果：指数寿命的混合必然呈递减风险率——Proschan (1963)[^proschan63] 用"机群故障率随时间下降"给出著名解释：脆弱的（快）模式先死，整体风险由长寿命模式主导。$v_t$ 下滑就是这个定理在优化上的重演；反过来，对 $d(t)$ 做谱反演（逆 Laplace，病态需正则化）可以做出"弛豫谱普查表"，在初始化时预报 run 长度。
- **统计物理**：宽谱弛豫系统的衰减普遍是幂律或拉伸指数；"层级约束弛豫"模型——上一层弛豫完成前下一层无法推进——几乎就是蒸馏的写照（Palmer et al. 1984[^palmer84]）：先对齐高频/一阶 token 边缘分布，再对齐句法，最后是低频细粒度结构。教师–学生在线学习理论早已推出同款定律：渐近误差由最小非零曲率特征值决定、呈幂律尾巴（Seung, Sompolinsky & Tishby 1992[^seung92]；Saad & Solla 1995[^saad-solla-95] 的"指数专业化阶段 + 慢亚模式弛豫"两段式）。
- **学习曲线/标度律**：幂律学习曲线 = 指数分量的谱和，已有正式理论（Hutter 2021[^hutter21]，其 $n^{-\beta}$ 框架与本节的 Laplace 变换同构，只是自变量从数据量换成时间）；LM 预训练的 step 维度幂律 $L(t)\propto t^{-\alpha}$ 与 optimization-limited regime 是同一现象的预训练版（Kaplan et al. 2020[^kaplan20]；谱分解见 Bahri et al. 2021[^bahri21]）；数据侧的量化/层级结构解释谱振幅为何衰减（Michaud et al. 2023[^michaud23]）。
- **频率原理/谱偏差**："网络按曲率从高到低依次学习"有直接的实验与理论文献（Rahaman et al. 2019[^rahaman19]、Xu et al. 2020[^xu20] 的 F-Principle）——本模型的吸收前沿就是这个现象在 (学生, 教师) 分布匹配问题上的表现。
- **OPD 机制近作**（论文 §8[^fu26] 自己引的线索，与本模型相容）："dense per-token target 在参数空间锁定一个 early-locked 低秩更新子空间"（Cai et al. 2026[^cai26]、Shen et al. 2026[^shen26]）——低秩锁定 = 刚性模式先被吃掉；"哪些 token 可学"（Armandpour et al. 2026[^armandpour26]、Wang et al. 2026[^wang26b]）——对应谱振幅 $a_i$ 在 token 上的分布。
- **KD 实践侧的平行经验**：Beyer et al. (2022)[^beyer22] 的结论是"好的教师需要耐心与一致性"——长蒸馏日程 + 强增广能稳定对齐；与本模型"长尾由谱决定、不应指望靠换数据急剧缩短"互为印证。

---

## 4. 待检验假设汇总

| #   | 假设                                  | 关键预言                                             | 检验                                     |
| --- | ------------------------------------- | ---------------------------------------------------- | ---------------------------------------- |
| H1  | 尾部是幂律/拉伸指数（谱混合）         | $v\propto1/t$；$v_{50}/v_{200}\approx4$              | 已被论文数据支持[^fu26]；D1 拟合分辨形态 |
| H2  | 存在噪声底 $\propto\eta/B$ + 表示缺口 | off-policy run 的平台高度随 $\eta$ 降、随 $B$ 升而降 | D2                                       |
| H3  | AdamW $\beta_2$ 记忆垫高尾部          | $\beta_2=0.95$ 或末期换优化器 → 尾巴变陡             | D3                                       |
| H4  | 残差质量随训练迁向低频 token          | 逐 token 残差按频率分层的质心下移                    | D4                                       |
| H5  | 谱可测且可预测 run                    | step-30 谱普查预测整条 $v_t$                         | D5                                       |
| H6  | 指数是"优化器×架构"性质               | 换 Muon/SOAP 后 $\alpha$ 改变                        | D6                                       |

---

## 5. 判别实验（按成本排序）

- **D1 尾巴形态拟合**：对 $d(t)$（建议同时拟合 $d^2$，噪声减半）分别拟合 $t^{-\alpha}$、$e^{-(t/\tau)^\gamma}$、$d_\infty+Ae^{-t/\tau}$ 三种形式并比较残差。三者含义：幂律=谱幂律+振幅衰减；拉伸指数=宽谱近均匀；floor+指数=单一主模式+底。
- **D2 底的标度**：off-policy run 扫 $\eta\in\{\tfrac12,1,2\}\times10^{-6}$ 与 $B\in\{64,256\}$：平台高度（平方距离）应 $\propto\eta/B$；不随 $\eta$ 动的那部分是表示缺口（查温度、tokenizer、top-$k$ 截断）。
- **D3 Adam 病态检查**：$\beta_2\in\{0.95,0.999\}$、$\epsilon$ 扫描、末期切 SGD-with-decay 或 SOAP，看 $v_t$ 尾部。
- **D4 直拍吸收前沿（最推荐，纯日志分析）**：每步把逐 token 残差 $|A_i|$ 按 token 一元频率分层（十等位），画各层均值随训练的轨迹；或按上下文熵分层。预言（H4）：残差质量从高频/简单结构迁向低频/细粒度。可仿照 `repetition/analyze_and_plot.py` 的离线分析模式，只需在 rollout 时多存一份逐 token $|A_i|$ 与 token id。
- **D5 谱普查 + 零训练预测**：在冻结 64 轨迹的目标上（batch 小，Hessian-vector 便宜），step 30 处用随机 Lanczos / 幂迭代（Ghorbani et al. 2019[^ghorbani19] 的方法）测 $\{\lambda_i\}$ 与 $\{c_i\}$，训练前用 $d(t)=\sum_i c_i^2e^{-2\eta\lambda_i t}$ 预测整条吸收曲线。命中 = 现象被解释，且 run 长度从此可在初始化时预报。
- **D6 优化器消融**：AdamW vs Muon vs SOAP，同样 300 步，看 $\alpha$（幂律指数）与 $v_{50}/v_{200}$ 变不变。直接检验 H6/"节奏是 OPD 算法性质"这句话里"算法"指的是哪一层。

---

## 6. 干预方案（按性价比排序）

**P1 吸收率闭环步长（首选，零额外成本）。** $v_t$ 本身就是 $\langle\lambda\rangle(t)/\eta$ 的在线读数，把它当控制器反馈：剩余曲率在衰减，稳定性上限（$\approx2/\lambda_{\mathrm{active}}$，edge-of-stability 直觉，Cohen et al. 2021[^cohen21]）在放松，所以 **lr 应随 $d$ 下降而上升——标准调度的时间反演**。论文 C.2[^fu26] 已证明曲线形状对 $\eta$ 不变，那么让 $\eta$ 随 $t$ 增长就是沿同一条曲线超线性前进（$u=\int\eta_t\,dt$）。

```python
# 每 K 步（K=5~10）：
v_hat = ema(v_measured, 0.2)  # 论文的几何窗口估计量
if v_hat < v_target:  # 落后于目标吸收率 → 提速
    eta = min(eta * (1 + kappa * (v_target / v_hat - 1)), eta_stab)
else:  # 达标 → 缓降，防噪声底抬升过快
    eta = max(eta * (1 - kappa * (1 - v_target / v_hat)), eta_min)
# 安全网：grad-clip（已存在，前 30 步本来就在 clip）、
#         单步 d 增幅检查（trust region on d）、验证指标回退即回退
# 建议：v_target ≈ 0.05–0.1，kappa ≈ 0.1，eta_stab 用 2×/4× lr 探针步校准
```

副作用：抬高噪声底（$\propto\eta$），必须配 P2；验证指标平台即停升。

**P2 把底压下去（方差工程）。** 末期 batch 加倍/四倍（Smith et al. 2018[^smith18] 的"别降 lr、升 batch"；McCandlish et al. 2018[^mccandlish18] 的 gradient noise scale）+ 权重 EMA / Polyak 平均（常数步长 + 迭代平均达到随机强凸的统计最优率，Dieuleveut, Flammarion & Bach 2017[^dieuleveut-17]；EMA 修复 LM step-scaling 的系统性偏差，Porian et al. 2024[^porian24]）。机制：把 $\lambda_{\mathrm{crit}}\propto\sigma/(c\sqrt{B})\cdot\sqrt{\eta}$ 往谱深处推，让原本被噪声挡住的慢模式变得可吸收。注意这个反直觉推论：**更大的 batch 不是让 run 变短，而是让你吃得更深**——可见的 run 可能变长，但同 step 数的最终 $d$ 更低。这为"流式数据抬高上限"的业界直觉提供了优化论注脚。

**P3 换几何：预条件/白化（唯一能改指数的杠杆）。** Muon（Jordan et al. 2024[^jordan24]；已验证到 3B/16B MoE 规模、同算力优于 AdamW，Liu et al. 2025[^liu25]）、Shampoo（Gupta, Koren & Singer 2018[^gupta18]）、SOAP（Vyas et al. 2024[^vyas24]）、K-FAC（Martens & Grosse 2015[^martens-grosse-15]）。机制：压缩 $\lambda_i$ 动态范围 = 把慢模式的弛豫速率拉平，直接改变 $\alpha$。预期行为：早期对数距离图显著变直（白化生效），尾部仍会弯——预条件子赢不了平坦方向上的信噪比底（Martens 2014[^martens14]）。D6 是它的判别版。

**P4 重塑损失：token 加权（保住不动点）。** "给 teacher 信号大的 token 加权"有隐藏陷阱：朴素加权会移动目标最优解——学生最终收敛到**加权 teacher** 而非 teacher。两种安全做法：

1. **退火到无偏**：$w_i(t)=1+\lambda(t) f(A_i)$，$\lambda(t)\to0$（早期加速、末期回到真目标）；
2. **重要性校正**：按加权方案的采样概率除回去，保持估计量无偏。

思路来源：Rho-1 的 Selective Language Modeling（Lin et al. 2024[^lin24]，但它接受改变目标）、focal loss 的调制思想（Lin et al. 2017[^lin17]）。相关证据：大量 token 无可学信号、目前只贡献噪声 $\sigma_i$（Armandpour et al. 2026[^armandpour26]、Wang et al. 2026[^wang26b]）；论文的 top-$k$ 估计器（Eq. 3）已经是这个方向的半个实现（截断尾部降方差）。预期：提高早期 $v$、缩短同质量 run，**不改变指数**——谱是参数化的，加权改变不了它。

**P5 trust region 内多 epoch 复用（预测无效，且论文已含对照组）。** "always off-policy" 冻结 64 轨迹的 run 就是无限 epoch 复用的极限，它照样走了 200 步（论文 §5.3[^fu26]）。复用攻击数据供给，瓶颈在吸收；复用只在"瓶颈是梯度估计方差"时有用——那时它等价于 P2 的方差缩减。单独做预期无效；与 P3 组合值得一试。此条立字为据，欢迎推翻。

**P6 把吸收预算变成一等公民（工程决策框架）。** $d(t)\sim t^{-\alpha}+d_\infty$ 意味着每步边际收益可预测地递减：给下游任务定 $d$ 预算（等价于 state-coverage / gap-recovery 目标），到点就停，把省下的算力花在 teacher 质量与状态多样性上（论文 §7[^fu26] 已证明两者才是上限变量）。数据注入与吸收工程是两个正交轴：前者抬天花板（覆盖 + 降 $\sigma$），后者抬斜率。

---

## 7. 局限与开放问题

- 线性化在 clip 失效（step 30）之后才严格成立；最初 30 步的动力学不归这个模型管。
- $d(t)=\sum_i a_i e^{-\eta\lambda_i t}$ 假设了各模式对 $|\log\text{-ratio}|$ 的贡献近似可加（符号混合为高阶小量）；$d^2$ 度量下该假设自动成立，这也是 D1 建议同时拟合 $d^2$ 的原因。
- on-policy 时 Hessian 随状态分布漂移、模式间有耦合；冻结状态实验说明它是二阶效应，但 D5 的谱预测应先在 off-policy 设置下验证。
- 推导五指出可吸收性阈值是估计论极限，但没有回答"为什么谱是幂律的"。本模型把它归约为架构 + 目标的性质；其起源是另一个开放问题（NTK 谱 Basri et al. 2020[^basri20]、谱偏差 Rahaman et al. 2019[^rahaman19] 与 Xu et al. 2020[^xu20]、数据侧量化结构 Michaud et al. 2023[^michaud23]、edge-of-stability 的曲率锐化 Cohen et al. 2021[^cohen21]）。D4 是连接优化侧与数据侧故事的最直接探针。
- H3（AdamW 软底）目前只是启发式分析 + 可证伪预测。

---

### 参考文献

[^fu26]: Z. Fu, B. He, Y. Zuo, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)（代码：[thunlp/OPD](https://github.com/thunlp/OPD)）

[^wang26a]: Y. Wang, Q. Yang, Z. Zeng, et al. Reinforcement Learning for Reasoning in Large Language Models with One Training Example. arXiv 2026.（编号见 [^fu26] 参考文献）

[^opd1-26]: Z. Fu, et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe (Part I). arXiv 2026. [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)

[^tml25]: Thinking Machines Lab (K. Lu). Connectionist: On-Policy Distillation. 博文 2025. [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)

[^song26]: M. Song, M. Zheng. A Survey of On-Policy Distillation for Large Language Models. arXiv 2026.（编号见 [^fu26] 参考文献）

[^gu24]: Y. Gu, L. Dong, F. Wei, M. Huang. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)

[^agarwal24]: R. Agarwal, N. Vieillard, Y. Zhou, et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD). ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

[^ross11]: S. Ross, G. Gordon, J. A. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)

[^amari98]: S. Amari. Natural Gradient Works Efficiently in Learning. Neural Computation 10(2):251–276, 1998.

[^amari00]: S. Amari, H. Nagaoka. Methods of Information Geometry. AMS/Oxford, 2000.

[^jacot18]: A. Jacot, F. Gabriel, C. Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. NeurIPS 2018. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)

[^arora19]: S. Arora, S. S. Du, W. Hu, Z. Li, R. Salakhutdinov, R. Wang. Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks. ICML 2019. [arXiv:1901.08584](https://arxiv.org/abs/1901.08584)

[^chizat19]: L. Chizat, E. Oyallon, F. Bach. On Lazy Training in Differentiable Programming. NeurIPS 2019.

[^ghorbani19]: B. Ghorbani, S. Krishnan, Y. Xiao. An Investigation into Neural Net Optimization via Hessian Eigenvalue Density. AISTATS 2019. [arXiv:1901.10159](https://arxiv.org/abs/1901.10159)

[^papyan20]: V. Papyan. The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD Training. IEEE Trans. Inf. Theory, 2020.

[^basri20]: R. Basri, J. Braunstein, N. Friedman, D. Kerdreux, J. Jacobs. The Spectrum of the Neural Tangent Kernel at Initialization. 2020.

[^yao07]: Y. Yao, L. Rosasco, A. Caponnetto. On Early Stopping in Gradient Descent Learning. Constructive Approximation 26:289–315, 2007.

[^caponnetto07]: A. Caponnetto, E. De Vito. Optimal Rates for the Regularized Least-Squares Algorithm. Foundations of Computational Mathematics 7:331–368, 2007.

[^dieuleveut16]: A. Dieuleveut, F. Bach. Nonparametric Stochastic Approximation with Large Step-Sizes. Annals of Statistics 44(4):1363–1399, 2016.

[^mandt17]: S. Mandt, M. Hoffman, D. Blei. Stochastic Gradient Descent as Approximate Bayesian Inference. JMLR 18(134):1–35, 2017. [arXiv:1704.04289](https://arxiv.org/abs/1704.04289)

[^kakade02]: S. Kakade. A Natural Policy Gradient. NeurIPS 2002.

[^schulman15]: J. Schulman, S. Levine, P. Abbeel, M. Jordan, P. Moritz. Trust Region Policy Optimization. ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)

[^schulman17]: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal Policy Optimization Algorithms. arXiv 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^tomar21]: M. Tomar, U. A. Mishra, A. Rajeswaran, L. Yang. Mirror Descent Policy Optimization. AAAI 2021.

[^martens14]: J. Martens. New Insights and Perspectives on the Natural Gradient Method. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^agarwal12]: A. Agarwal, P. L. Bartlett, P. Ravikumar, M. J. Wainwright. Information-Theoretic Lower Bounds on the Oracle Complexity of Stochastic Convex Optimization. IEEE Trans. Inf. Theory 58(5):3235–3249, 2012.

[^bach-moulines-13]: F. Bach, E. Moulines. Non-asymptotic Analysis of Stochastic Approximation Algorithms for Machine Learning. NeurIPS 2013.

[^dieuleveut-17]: A. Dieuleveut, N. Flammarion, F. Bach. Harder, Better, Faster, Stronger Convergence Rates for Least-Squares Regression. JMLR 18(101):1–52, 2017.

[^reddi18]: S. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. ICLR 2018.

[^proschan63]: F. Proschan. Theoretical Explanation of Observed Decreasing Failure Rate. Technometrics 5(3):375–383, 1963.

[^palmer84]: R. G. Palmer, D. L. Stein, E. Abrahams, P. W. Anderson. Models of Hierarchically Constrained Relaxation. Phys. Rev. Lett. 53(10):958–961, 1984.

[^seung92]: H. S. Seung, H. Sompolinsky, N. Tishby. Statistical Mechanics of Learning from Examples. Phys. Rev. A 45:6056–6091, 1992.

[^saad-solla-95]: D. Saad, S. Solla. On-Line Learning in Soft Committee Machines. Phys. Rev. Lett. 74:4337, 1995.

[^hutter21]: M. Hutter. Learning Curve Theory. arXiv 2021. [arXiv:2102.04074](https://arxiv.org/abs/2102.04074)

[^kaplan20]: J. Kaplan, S. McCandlish, T. Henighan, et al. Scaling Laws for Neural Language Models. arXiv 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

[^bahri21]: Y. Bahri, E. Dyer, J. Kaplan, J. Lee, U. Sharma. Explaining Neural Scaling Laws. arXiv 2021（PNAS 2024）. [arXiv:2102.06701](https://arxiv.org/abs/2102.06701)

[^michaud23]: E. J. Michaud, Z. Liu, O. Girit, C. Lialin, R. G. Baraniuk. The Quantization Model of Neural Scaling. NeurIPS 2023.

[^rahaman19]: N. Rahaman, A. Baratin, D. Arpit, et al. On the Spectral Bias of Neural Networks. ICML 2019. [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)

[^xu20]: Z. Q. J. Xu, Y. Zhang, T. Luo. Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks. Communications in Computational Physics 28(5):1746–1767, 2020. [arXiv:1901.06523](https://arxiv.org/abs/1901.06523)

[^cai26]: Y. Cai, D. Cao, L. Lin, et al. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^shen26]: Z. Shen, Y. Li, Q. Yin, et al. On the Geometry of On-Policy Distillation. arXiv 2026. [arXiv:2606.07082](https://arxiv.org/abs/2606.07082)

[^armandpour26]: M. Armandpour, F. Ilhan, D. Harrison, et al. Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why. arXiv 2026. [arXiv:2605.10889](https://arxiv.org/abs/2605.10889)

[^wang26b]: Y. Wang, S. Lu, Y. Gu, et al. Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation. arXiv 2026. [arXiv:2605.26844](https://arxiv.org/abs/2605.26844)

[^beyer22]: L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, A. Kolesnikov. Knowledge Distillation: A Good Teacher Is Patient and Consistent. CVPR 2022. [arXiv:2106.05237](https://arxiv.org/abs/2106.05237)

[^cohen21]: J. Cohen, A. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar. Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR 2021. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)

[^smith18]: S. L. Smith, P.-J. Kindermans, C. Ying, Q. V. Le. Don't Decay the Learning Rate, Increase the Batch Size. ICLR 2018. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489)

[^mccandlish18]: S. McCandlish, J. Kaplan, D. Amos, et al. An Empirical Model of Large-Batch Training. arXiv 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

[^porian24]: T. Porian, M. Wortsman, J. M. Hernández-Lobato, A. Rives, R. B. Grosse. Resolving Discrepancies in Compute-Optimal Scaling of Language Models. NeurIPS 2024.

[^jordan24]: K. Jordan, et al. Muon: Momentum Orthogonalized by Newton–Schulz. 开源优化器 2024. [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)

[^liu25]: J. Liu, J. Su, X. Yao, et al. Muon is Scalable for LLM Training. arXiv 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

[^gupta18]: V. Gupta, T. Koren, Y. Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568)

[^vyas24]: N. Vyas, D. Morwani, R. Zhao, et al. SOAP: Improving and Stabilizing Shampoo using Adam for Deep LLM Training. arXiv 2024. [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)

[^martens-grosse-15]: J. Martens, R. Grosse. Optimizing Neural Networks with Kronecker-factored Approximate Curvature. ICML 2015. [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)

[^lin24]: Z. Lin, Z. Gou, Y. Gong, et al. Rho-1: Not All Tokens Are What You Need. NeurIPS 2024. [arXiv:2404.07965](https://arxiv.org/abs/2404.07965)

[^lin17]: T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. Focal Loss for Dense Object Detection. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
