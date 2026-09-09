# 学习率、Batch Size 与自适应优化器：六篇科学空间文章的结论与推导

**日期**：2026-09-08<br>
**定位**：科学空间 6 篇文章的统一阅读笔记，重点记录可复用的推导和它们成立所需的假设。

本文整理本地保存的 6 篇科学空间文章，主题集中在学习率与 Batch Size 的关系、Muon、EMA/动量、Adam 的 $\beta_1=\beta_2$，以及从 Hessian 近似理解自适应学习率。

这不是把原文逐段改写，而是抽取一条共同的数学主线，并把每个结论标记为：

- **恒等式/直接推导**：在给定定义和假设下可以直接验证；
- **二阶局部近似**：依赖小步长和局部 Hessian；
- **平均场近似**：把非线性函数的期望用低阶矩替代；
- **结构假设**：例如独立分量、各向同性噪声、特定 Hessian 形式或梯度缓变。

因此，文中的“证明”大多是“在明确假设下的推导”，不等同于对真实深度网络全程训练的全局收敛定理。

## 摘要

这 6 篇文章可以压缩成一条推理链：

1. 局部二阶 Taylor 展开把最优学习率写成更新方向一阶矩与二阶矩的比值；
2. Batch Size 通过梯度噪声协方差进入二阶矩，因此 SGD 呈现“线性到饱和”；
3. SignSGD 和 Muon 的归一化结构把小 Batch 渐近改成平方根律，并在一般 Hessian 下可能出现 Surge；
4. EMA/动量把有效噪声规模放大约 $\frac{1+\beta_1}{1-\beta_1}$ 倍；
5. $\beta_1=\beta_2$ 的主要理论理由是同权重均值—方差分解带来的有界更新；
6. Adam 的 $\sqrt{v}$ 可以在局部统计假设下理解为对角 Hessian 的尺度估计。

这些结论是局部近似、平均场近似或结构假设下的推导，不是对真实深度网络全程训练的无条件定理。

## 结论先行

| 问题                                 | 可复用结论                                                                           | 主要限制                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| SGD 的学习率如何随 Batch Size 变化？ | $\eta_B^*=\eta_{\max}/(1+\mathcal B_{\mathrm{noise}}/B)$；小 Batch 近似线性，大 Batch 饱和。              | 局部二次模型和正的二次项。                             |
| SignSGD/Muon 为什么常见平方根律？    | 平均场下 $\rho_B=(1+\mathcal B_{\mathrm{simple}}/B)^{-1/2}$，小 Batch 时 $\eta_B^*\propto\sqrt{B}$。 | 分量独立、矩阵各向同性和平均场闭合。                   |
| Surge 从哪里来？                     | 学习率分母含 $D/\rho+C\rho$；当 Hessian 的非对角/方向项足够大时先增后减。           | 这是优化器假设与真实 Hessian 不匹配的信号。            |
| 动量改变了什么？                     | 稳态独立噪声下，$B_{\mathrm{eff}}=B(1+\beta_1)/(1-\beta_1)$。                                     | 梯度缓变、跨步噪声独立；真实序列会有相关性。           |
| $\beta_1=\beta_2$ 为什么稳定？           | 同一组 EMA 权重给出 $\hat v_t=\hat m_t^2+\hat\sigma_t^2$，从而 $\lvert u_t\rvert\le1$。        | 这是稳定性结论，不等于所有任务上的泛化最优。           |
| Adam 的 $v$ 是否像 Hessian？         | 在终点附近、对角 Hessian、参数扰动近似各向同性时，$\sqrt{v}$ 可近似 Hessian 尺度。    | 瞬时 Gauss-Newton、长期 Hessian 近似不是同一个恒等式。 |

## 1. 阅读范围

| 编号            | 文章                                      | 本文对应主题                            |
| --------------- | ----------------------------------------- | --------------------------------------- |
| 11260 [^k11260] | 重新思考学习率与 Batch Size（一）：现状   | 二阶最优学习率、SGD、数据效率           |
| 11280 [^k11280] | 重新思考学习率与 Batch Size（二）：平均场 | SignSGD/SoftSignSGD 的平均场计算、Surge |
| 11285 [^k11285] | 重新思考学习率与 Batch Size（三）：Muon   | 矩阵 $\operatorname{msign}$ 与 Muon 的 Batch Size 规律   |
| 11301 [^k11301] | 重新思考学习率与 Batch Size（四）：EMA    | 动量等效放大 Batch Size、Adam 的 Surge  |
| 11593 [^k11593] | Adam 优化器的最优超参数是 $\beta_1=\beta_2$？   | 在线矩估计、信噪比感知、稳定性          |
| 10588 [^k10588] | 从 Hessian 近似看自适应学习率优化器       | Adam 作为对角 Newton 法的解释           |

## 2. 统一记号与总母题

本节抽取 11260、11280、11285 和 11301 中共同使用的二阶分析框架。[^k11260][^k11280][^k11285][^k11301]

设总体损失为 $L(w)$，参数为 $w$，总体梯度和 Hessian 分别为

$$
g=\nabla L(w),\qquad H=\nabla^2 L(w).
$$

Batch Size 为 $B$ 时，随机 Batch 梯度记为 $\tilde g_B$。对独立同分布样本取平均，有

$$
\mathbb E[\tilde g_B]=g,
\qquad
\operatorname{Cov}(\tilde g_B)=\frac{\Sigma}{B}.
$$

把一次更新写成

$$
w^+=w-\eta\tilde\varphi_B,
$$

其中 $\tilde\varphi_B$ 是优化器真正使用的更新方向：SGD 中它是 $\tilde g_B$，SignSGD 中它是 $\operatorname{sign}(\tilde g_B)$，Muon 中它是矩阵 $\operatorname{msign}(\tilde G_B)$。

### 2.1 二阶最优步长引理

对 $L(w-\eta\tilde\varphi_B)$ 做二阶 Taylor 展开：

$$
L(w-\eta\tilde\varphi_B)
\approx
L(w)-\eta\tilde\varphi_B^\top g
+\frac12\eta^2\tilde\varphi_B^\top H\tilde\varphi_B.
$$

对随机性取期望，并利用 $x^\top Hx=\operatorname{tr}(xx^\top H)$：

$$
\mathbb E[L(w-\eta\tilde\varphi_B)]
\approx
L(w)-\eta a_B+\frac12\eta^2 c_B,
$$

其中

$$
a_B=\mathbb E[\tilde\varphi_B]^\top g,
\qquad
c_B=\operatorname{tr}\!\left(\mathbb E[\tilde\varphi_B\tilde\varphi_B^\top]H\right).
$$

若 $a_B>0$ 且 $c_B>0$，这是关于 $\eta$ 的凸二次函数。求导得

$$
\frac{\mathrm d}{\mathrm d\eta}
\left(-\eta a_B+\frac12\eta^2c_B\right)
=-a_B+\eta c_B=0.
$$

所以

$$
\boxed{\eta_B^*\approx\frac{a_B}{c_B}}
$$

并且在这个局部二次模型中的最大平均损失下降量为

$$
\boxed{\overline{\Delta L}_B
\approx\frac{a_B^2}{2c_B}}.
$$

这是 11260、11280、11285 和 11301 的共同出发点。不同优化器的差别，集中在如何计算

$$
\mathbb E[\tilde\varphi_B],
\qquad
\mathbb E[\tilde\varphi_B\tilde\varphi_B^\top].
$$

### 2.2 这条引理的适用边界

1. $\eta_B^*$ 是当前点附近、平均意义下的最优步长，不是整个训练过程的全局最优常数。
2. $H$ 必须在这一步附近足够稳定，且二阶余项不能主导。
3. $c_B>0$ 是关键条件。Hessian 非正定、步长过大或轨迹远离局部区域时，二次模型可能不再可靠。
4. 要把单步损失下降转化为总步数和总数据量，还要额外假设局部统计量在训练过程中变化不太快。

## 3. 11260：SGD 的学习率与 Batch Size

本节对应原文 11260。[^k11260]

### 3.1 SGD 的精确矩计算

SGD 取 $\tilde\varphi_B=\tilde g_B$。由协方差定义，

$$
\begin{aligned}
\mathbb E[\tilde g_B\tilde g_B^\top]
&=\operatorname{Cov}(\tilde g_B)
+\mathbb E[\tilde g_B]\mathbb E[\tilde g_B]^\top\\
&=\frac{\Sigma}{B}+gg^\top.
\end{aligned}
$$

因此

$$
a_B=g^\top g,
$$

$$
c_B=\operatorname{tr}\left(\left(gg^\top+\frac{\Sigma}{B}\right)H\right)
=g^\top Hg+\frac{\operatorname{tr}(\Sigma H)}{B}.
$$

代入二阶最优步长引理：

$$
\eta_B^*
\approx
\frac{g^\top g}{g^\top Hg+\operatorname{tr}(\Sigma H)/B}.
$$

定义

$$
\eta_{\max}=\frac{g^\top g}{g^\top Hg},
\qquad
\mathcal B_{\mathrm{noise}}
=\frac{\operatorname{tr}(\Sigma H)}{g^\top Hg},
$$

可得核心结论：

$$
\boxed{
\eta_B^*\approx
\frac{\eta_{\max}}{1+\mathcal B_{\mathrm{noise}}/B}}
$$

### 3.2 解释：线性区和饱和区

- 当 $B$ 很小时，$B\ll\mathcal B_{\mathrm{noise}}$，有

$$
\eta_B^*\approx
\eta_{\max}\frac{B}{\mathcal B_{\mathrm{noise}}}.
$$

    即小 Batch 区域近似线性缩放。

- 当 $B$ 很大时，$B\gg\mathcal B_{\mathrm{noise}}$，有

$$
\eta_B^*\to\eta_{\max}.
$$

    学习率趋于上界，而不是可以无限按线性律增加。

$\mathcal B_{\mathrm{noise}}$ 是 Hessian 加权的噪声强度。LLM 中通常无法直接计算完整 $H$，于是使用

$$
\mathcal B_{\mathrm{simple}}
=\frac{\operatorname{tr}(\Sigma)}{g^\top g}
$$

作为简化统计量。它在 $H$ 近似为单位阵的倍数时与 $\mathcal B_{\mathrm{noise}}$ 一致；一般情况下只是启发式近似，可以理解为噪声能量与梯度信号能量之比。

### 3.3 数据量—步数关系

将 $\eta_B^*$ 代回期望损失，可得

$$
\overline{\Delta L}_B
\approx
\frac{\Delta L_{\max}}
{1+\mathcal B_{\mathrm{noise}}/B},
\qquad
\Delta L_{\max}
=\frac{(g^\top g)^2}{2g^\top Hg}.
$$

在局部统计量固定的假设下，令 $T_{\min}$ 为无穷大 Batch 的最少步数，则有限 Batch 达到相同目标大约需要

$$
T=\left(1+\frac{\mathcal B_{\mathrm{noise}}}{B}\right)T_{\min}.
$$

总样本消耗量为

$$
E=BT=(B+\mathcal B_{\mathrm{noise}})T_{\min}.
$$

形式上定义

$$
E_{\min}=\mathcal B_{\mathrm{noise}}T_{\min},
$$

就得到经典的双曲关系：

$$
\boxed{
\left(\frac{T}{T_{\min}}-1\right)
\left(\frac{E}{E_{\min}}-1\right)=1
}.
$$

验证很直接：

$$
\frac{T}{T_{\min}}-1=\frac{\mathcal B_{\mathrm{noise}}}{B},
\qquad
\frac{E}{E_{\min}}-1=\frac{B}{\mathcal B_{\mathrm{noise}}},
$$

两者相乘即为 1。这个结论比具体的 $\eta(B)$ 形式更稳健，但仍依赖“不同 Batch 只改变噪声规模、局部下降效率近似可累加”的假设。

## 4. 11280：平均场、SignSGD 与 Surge

本节对应原文 11280。[^k11280]

### 4.1 平均场的核心近似

对标量随机变量 $x$，SignSGD 的核心量是

$$
\mathbb E[\operatorname{sign}(x)]
=\mathbb E\left[\frac{x}{\sqrt{x^2}}\right].
$$

平均场近似把非线性函数中的随机量替换为低阶矩：

$$
\boxed{
\mathbb E[\operatorname{sign}(x)]
\approx
\frac{\mathbb E[x]}{\sqrt{\mathbb E[x^2]}}
}.
$$

若 $\mathbb E[x]=g$、$\operatorname{Var}(x)=\sigma^2/B$，则

$$
\mathbb E[\operatorname{sign}(x)]
\approx
\frac{g}{\sqrt{g^2+\sigma^2/B}}.
$$

这一步不是一般意义下的恒等式，也不需要正态分布才能写出，但它需要额外的平均场闭合假设。若 $x$ 服从正态分布，精确结果会涉及 $\operatorname{erf}$，平均场形式可以视为一种简单的低阶近似。

### 4.2 SignSGD 的一阶和二阶矩

逐分量写 $\sigma_i^2=\Sigma_{ii}$：

$$
\mathbb E[\tilde\varphi_{B,i}]
\approx
\frac{g_i}{\sqrt{g_i^2+\sigma_i^2/B}}.
$$

为了得到只依赖标量 $B$ 的形式，再用一次平均场近似，把不同坐标的 $\sigma_i^2/g_i^2$ 替换成统一的信噪比倒数：

$$
\mathcal B_{\mathrm{simple}}
=\frac{\operatorname{tr}(\Sigma)}{g^\top g},
\qquad
\rho_B=\left(1+\frac{\mathcal B_{\mathrm{simple}}}{B}\right)^{-1/2},
$$

并记 $s=\operatorname{sign}(g)$，则

$$
\mathbb E[\tilde\varphi_B]\approx\rho_Bs.
$$

若进一步假设不同坐标的符号更新独立，则对 $i\ne j$ 有

$$
\mathbb E[\tilde\varphi_{B,i}\tilde\varphi_{B,j}]
\approx
\mathbb E[\tilde\varphi_{B,i}]
\mathbb E[\tilde\varphi_{B,j}],
$$

而对角线上 $\operatorname{sign}(x)^2=1$。因此

$$
M_B:=\mathbb E[\tilde\varphi_B\tilde\varphi_B^\top]
\approx
\rho_B^2ss^\top+(1-\rho_B^2)I.
$$

### 4.3 SignSGD 的最优学习率

设

$$
S_1=\sum_i\lvert g_i\rvert,
\qquad
D=\operatorname{tr}(H),
\qquad
A=s^\top Hs,
\qquad
C=A-D.
$$

则

$$
a_B=(\rho_Bs)^\top g=\rho_BS_1,
$$

并且

$$
c_B=\operatorname{tr}(M_BH)
=\rho_B^2A+(1-\rho_B^2)D
=D+\rho_B^2C.
$$

所以

$$
\boxed{
\eta_B^*
\approx
\frac{\rho_BS_1}{D+\rho_B^2C}
=\frac{S_1}{D/\rho_B+\rho_BC}
}.
$$

当 $B\ll\mathcal B_{\mathrm{simple}}$ 时，$\rho_B$ 近似 $\sqrt{B/\mathcal B_{\mathrm{simple}}}$，于是

$$
\eta_B^*\propto\sqrt B.
$$

这解释了 SignSGD 在小 Batch 区域更接近平方根缩放，而不是 SGD 的线性缩放。

### 4.4 Surge 条件

关于 $\rho_B$ 的分母是

$$
f(\rho)=\frac D\rho+C\rho.
$$

因为 $\rho$ 随 $B$ 单调增加，学习率是否反常由 $f$ 是否先降后升决定。

- 若 $C\le0$，则 $f'(\rho)=-D/\rho^2+C<0$，学习率随 Batch Size 单调增加。
- 若 $C>0$，分母的驻点为

$$
\rho^*=\sqrt{\frac DC}.
$$

    只有 $\rho^*<1$，即 $C>D$，该驻点才落在实际区间 $(0,1)$ 内。此时学习率先增加后减少，出现 Surge。

临界 Batch Size 可显式写成

$$
\boxed{
B_{\mathrm{crit}}
=\frac{D}{C-D}\mathcal B_{\mathrm{simple}}
\qquad(C>D).
}
$$

这里 $H$ 正定只保证 $D>0$ 和 $A>0$，并不保证 $C\le0$，所以 Surge 并不与局部凸性矛盾。

### 4.5 为什么学习率会 Surge，但损失下降不 Surge

利用 $\overline{\Delta L}=a_B^2/(2c_B)$：

$$
\overline{\Delta L}_B
\approx
\frac{\rho_B^2S_1^2}{2(D+\rho_B^2C)}.
$$

将 $\rho_B^2=1/(1+\mathcal B_{\mathrm{simple}}/B)$ 代入并整理：

$$
\overline{\Delta L}_B
\approx
\frac{\Delta L_{\max}}
{1+\mathcal B_{\mathrm{noise}}/B},
$$

其中

$$
\Delta L_{\max}=\frac{S_1^2}{2(D+C)}=\frac{S_1^2}{2A},
\qquad
\mathcal B_{\mathrm{noise}}
=\frac{\mathcal B_{\mathrm{simple}}D}{D+C}.
$$

因此，学习率是 $a_B/c_B$，受到更新方向和 Hessian 的非齐次影响；损失下降量是 $a_B^2/(2c_B)$，分子和分母中的同一缩放因子会部分抵消。于是可能出现：**最优学习率在大 Batch 端下降，但每步平均损失下降仍随 Batch 增加而增加**。

### 4.6 一般预条件器的解释

把一般更新写成

$$
\tilde\varphi_B=\tilde H_B^{-1}\tilde g_B,
$$

其中 $\tilde H_B$ 是随机半正定矩阵。令 $\hat H=\mathbb E[\tilde H_B]$，在平均场和近似独立假设下：

$$
\mathbb E[\tilde\varphi_B]
\approx \hat H^{-1}g,
$$

$$
\mathbb E[\tilde\varphi_B\tilde\varphi_B^\top]
\approx
\hat H^{-1}
\left(gg^\top+\frac{\Sigma}{B}\right)
\hat H^{-1}.
$$

于是

$$
\overline{\Delta L}_B
\approx
\frac12
\frac{(g^\top\hat H^{-1}g)^2}
{g^\top\hat H^{-1}H\hat H^{-1}g
+\operatorname{tr}(\Sigma\hat H^{-1}H\hat H^{-1})/B}.
$$

若 $\hat H\approx f(B)G$，其中 $f(B)$ 是标量，$f$ 在上式中会完全约掉，所以损失下降量仍呈现

$$
\frac{\text{常数}}{1+\text{常数}/B}
$$

的形式；但 $\eta_B^*$ 对 $\hat H$ 不是齐次的，因而不必具有统一的 Batch Size 规律。

这给出一个重要区分：**优化器改变了最优步长的表现，但在一定的齐次预条件假设下，不一定改变每步平均效率的渐近形式。**

### 4.7 SoftSignSGD：保留 $\epsilon$ 的广义形式

令

$$
\operatorname{softsign}_\epsilon(x)=\frac{x}{\sqrt{x^2+\epsilon^2}}.
$$

同样应用平均场近似，逐分量有

$$
\begin{aligned}
\mathbb E[\operatorname{softsign}_\epsilon(\tilde g_{B,i})]
&\approx
\frac{g_i}{\sqrt{g_i^2+\sigma_i^2/B+\epsilon^2}}\\
&=\frac{\operatorname{softsign}_\epsilon(g_i)}
{\sqrt{1+\frac{\sigma_i^2}{(g_i^2+\epsilon^2)B}}}.
\end{aligned}
$$

再把坐标相关的噪声比替换为统一标量，可取

$$
\mathcal B_{\mathrm{simple},\epsilon}
\approx
\frac{\operatorname{tr}(\Sigma)}{g^\top g+N\epsilon^2},
\qquad
\rho_B=\left(1+\frac{\mathcal B_{\mathrm{simple},\epsilon}}{B}\right)^{-1/2}.
$$

记 $\nu_i=\operatorname{softsign}_\epsilon(g_i)$，则

$$
\mathbb E[\tilde\varphi_B]\approx\rho_B\nu.
$$

在分量独立下，非对角二阶矩约为 $\rho_B^2\nu_i\nu_j$；对角线上则用

$$
\mathbb E[\operatorname{softsign}_\epsilon(\tilde g_{B,i})^2]
\approx
\frac{g_i^2+\sigma_i^2/B}
{g_i^2+\sigma_i^2/B+\epsilon^2}.
$$

用同一个标量近似后，矩阵形式可写为

$$
\mathbb E[\tilde\varphi_B\tilde\varphi_B^\top]
\approx
\rho_B^2\nu\nu^\top+(1-\rho_B^2)I.
$$

因此

$$
\boxed{
\eta_B^*
\approx
\frac{\rho_B\sum_i\nu_i g_i}
{\sum_iH_{ii}+\rho_B^2\left(\sum_{i,j}\nu_i\nu_jH_{ij}-\sum_iH_{ii}\right)}
}.
$$

它保留了 SignSGD 的 Batch Size 函数结构，同时把 $\epsilon$ 纳入信号尺度。特别地，$\lvert g_i\rvert\ll\epsilon$ 的坐标不再是纯符号更新，而回到近似线性的软符号区。

### 4.8 平均场近似什么时候较可信

对 $f(x)=x/\sqrt{x^2}$ 做平均场，相当于把分母中的 $x^2$ 替换成其均值 $\mu^2+\sigma^2$。若把

$$
x^2=\mu^2+\sigma^2+\delta,
\qquad
\mathbb E[\delta]=0,
$$

在分母处展开，则第一项给出平均场结果，下一项由三阶矩控制：

$$
\frac12\frac{\mathbb E[x^3]-(\mu^3+\mu\sigma^2)}
{(\mu^2+\sigma^2)^{3/2}}.
$$

对正态 $x\sim\mathcal N(\mu,\sigma^2)$，$\mathbb E[x^3]=\mu^3+3\mu\sigma^2$，于是该误差尺度化为

$$
\frac{\mu\sigma^2}{(\mu^2+\sigma^2)^{3/2}}.
$$

令 $r=\sigma^2/\mu^2$，其绝对值为 $r/(1+r)^{3/2}$，最大值在 $r=2$ 处取得，约为 $0.3849$；当信号远强于噪声或远弱于噪声时都趋近于 0。这不是严格误差上界，但说明平均场误差的主要来源是中等信噪比区间以及更高阶矩/重尾性。

## 5. 11285：Muon 为什么得到 SignSGD 式规律

本节对应原文 11285。[^k11285]

### 5.1 矩阵设置

令参数和梯度为 $W,G\in\mathbb R^{n\times m}$，且 $n\ge m$。Muon 的矩阵符号近似写成

$$
\operatorname{msign}(G)=G(G^\top G)^{-1/2}.
$$

假设 Batch 梯度每个元素的噪声独立、方差为 $\sigma^2/B$。这里把完整四阶协方差张量压缩成一个标量方差；对应的噪声总能量是 $nm\sigma^2/B$。

### 5.2 平均场计算 $\mathbb E[\operatorname{msign}(\tilde G_B)]$

和 SignSGD 一样，先把矩阵函数的期望近似移入函数：

$$
\begin{aligned}
\mathbb E[\operatorname{msign}(\tilde G_B)]
&=\mathbb E\left[\tilde G_B
(\tilde G_B^\top\tilde G_B)^{-1/2}\right]\\
&\approx
G\left(\mathbb E[\tilde G_B^\top\tilde G_B]\right)^{-1/2}.
\end{aligned}
$$

对 $i,j$ 分量计算：

$$
\mathbb E[\tilde G_B^\top\tilde G_B]
=G^\top G+\frac{n\sigma^2}{B}I_m.
$$

原因是对角项每一列有 $n$ 个噪声元素，额外贡献 $n\sigma^2/B$；非对角项的独立零均值噪声交叉项期望为零。

因此

$$
\mathbb E[\operatorname{msign}(\tilde G_B)]
\approx
G\left(G^\top G+\frac{n\sigma^2}{B}I_m\right)^{-1/2}.
$$

再作各向同性近似

$$
G^\top G\approx\frac{\operatorname{tr}(G^\top G)}{m}I_m,
$$

就得到

$$
\mathbb E[\operatorname{msign}(\tilde G_B)]
\approx
\operatorname{msign}(G)\rho_B,
$$

其中

$$
\rho_B=\left(1+\frac{\mathcal B_{\mathrm{simple}}}{B}\right)^{-1/2},
\qquad
\boxed{
\mathcal B_{\mathrm{simple}}
=\frac{mn\sigma^2}{\operatorname{tr}(G^\top G)}
=\frac{mn\sigma^2}{\lVert G\rVert_F^2}
}.
$$

最后一个等号使用了 $\operatorname{tr}(G^\top G)=\lVert G\rVert_F^2$。这也是把 $G$ 展平成向量后得到的噪声—信号比。原文该处的末项写作 $\lVert G\rVert_F$，按定义应理解为 Frobenius 范数的平方。

### 5.3 特定 Hessian 结构下的最优学习率

从 Newton 视角，Muon 的预条件结构对应于一个线性 Hessian 算子满足

$$
H X=\eta_{\max}^{-1}X(G^\top G)^{1/2}.
$$

由于 $\tilde\Phi_B=\operatorname{msign}(\tilde G_B)$ 满足

$$
\tilde\Phi_B^\top\tilde\Phi_B=I_m,
$$

有

$$
\begin{aligned}
\operatorname{tr}(\tilde\Phi_B^\top H\tilde\Phi_B)
&=\eta_{\max}^{-1}
\operatorname{tr}\left(
\tilde\Phi_B^\top\tilde\Phi_B(G^\top G)^{1/2}
\right)\\
&=\eta_{\max}^{-1}\operatorname{tr}\left((G^\top G)^{1/2}\right),
\end{aligned}
$$

这是与 Batch 噪声无关的常数。同时

$$
\operatorname{tr}\left(\mathbb E[\tilde\Phi_B]^\top G\right)
\approx
\rho_B\operatorname{tr}\left(\operatorname{msign}(G)^\top G\right).
$$

分子和分母中的 trace 因子相消，得到

$$
\boxed{
\eta_B^*\approx
\frac{\eta_{\max}}
{\sqrt{1+\mathcal B_{\mathrm{simple}}/B}}
}.
$$

所以，在这个匹配的 Hessian 结构下，Muon 与 SignSGD 的 Batch Size 规律一致：小 Batch 近似平方根缩放，大 Batch 趋于 $\eta_{\max}$。

### 5.4 结论的边界

Muon 的结论依赖矩阵噪声近似各向同性、$G^\top G$ 近似标量阵，以及特定的 Hessian 算子。对一般 Hessian，$\eta_B^*$ 也可能出现类似 SignSGD 的 Surge。文章的核心洞见不是“Muon 永远没有 Surge”，而是：**在匹配的矩阵几何下，矩阵符号化改变了更新方向，却没有改变这个标量噪声缩放律。**

## 6. 11301：EMA、动量与 Adam 的 Batch Size 规律

本节对应原文 11301。[^k11301]

为避免与 Adam 的 $\beta_1,\beta_2$ 混淆，以下用 $\rho_B$ 表示 Batch Size 因子，并定义

$$
\kappa=\frac{1-\beta_1}{1+\beta_1}.
$$

### 6.1 SGDM：动量等效放大 Batch Size

SGDM 的动量为

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)\tilde g_{B,t}.
$$

展开（取 $m_0=0$）：

$$
m_t=(1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}\tilde g_{B,s}.
$$

假设不同时间的 Batch 噪声独立，且训练进入稳定区后 $g_s$ 和 $\Sigma_s$ 变化缓慢。均值为

$$
\mathbb E[m_t]
\approx(1-\beta_1^t)g_t
\to g_t.
$$

协方差由独立噪声的方差可加性给出：

$$
\begin{aligned}
\operatorname{Cov}(m_t)
&\approx
\frac{(1-\beta_1)^2}{B}
\sum_{s=1}^t\beta_1^{2(t-s)}\Sigma_t\\
&=\frac{(1-\beta_1)^2(1-\beta_1^{2t})}
{(1-\beta_1^2)B}\Sigma_t\\
&\xrightarrow[t\to\infty]{}
\frac{1-\beta_1}{1+\beta_1}\frac{\Sigma_t}{B}
=\frac{\kappa\Sigma_t}{B}.
\end{aligned}
$$

也就是说，动量把噪声方差中的 $1/B$ 换成 $\kappa/B$。代入 SGD 的推导：

$$
\boxed{
\eta_B^*
\approx
\frac{\eta_{\max}}
{1+\kappa\mathcal B_{\mathrm{noise}}/B}
}.
$$

定义等效 Batch Size

$$
\boxed{
B_{\mathrm{eff}}
=\frac{B}{\kappa}
=B\frac{1+\beta_1}{1-\beta_1}
}.
$$

即可把 SGDM 看成用 EMA 低成本地降低梯度噪声。

### 6.2 SignSGDM 与 Muon

SignSGDM 对 $m_t$ 取符号。平均场给出

$$
\mathbb E[\operatorname{sign}(m_t)]
\approx
\frac{g_t}{\sqrt{g_t^2+\kappa\sigma_t^2/B}}
\approx
\frac{\operatorname{sign}(g_t)}
{\sqrt{1+\kappa\mathcal B_{\mathrm{simple}}/B}}.
$$

这就是 SignSGD 把 $B$ 替换为 $B_{\mathrm{eff}}$。因此，在同样的近似下，动量对 SignSGD 和 Muon 的作用都可以概括为

$$
\boxed{
B\longmapsto B\frac{1+\beta_1}{1-\beta_1}
}.
$$

该结论需要梯度噪声在时间上独立、梯度和方差缓变；它不是说动量真的增加了硬件 Batch，而是说它减少了进入最优步长公式的有效噪声。

### 6.3 Adam 的一阶矩和二阶矩

在稳态近似 $t\to\infty$、忽略 bias correction 和 $\epsilon$ 时：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)\tilde g_{B,t},
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)\tilde g_{B,t}^2.
$$

上一节给出

$$
\mathbb E[m_t]\approx g_t,
\qquad
\mathbb E[m_t^2]\approx g_t^2+\frac{\kappa\sigma_t^2}{B}.
$$

而二阶 EMA 的均值在梯度缓变时为

$$
\mathbb E[v_t]
\approx
g_t^2+\frac{\sigma_t^2}{B}.
$$

因此 Adam 的更新量 $u_t=m_t/\sqrt{v_t}$ 满足

$$
\mathbb E[u_t]
\approx
\frac{g_t}{\sqrt{g_t^2+\sigma_t^2/B}}
\approx \rho_B\operatorname{sign}(g_t),
$$

这解释了为什么 SignSGD 可以作为 Adam 的一阶近似；但二阶矩不同：

$$
\boxed{
\mathbb E[u_t^2]
\approx
\frac{g_t^2+\kappa\sigma_t^2/B}
{g_t^2+\sigma_t^2/B}
}.
$$

所以 Adam 的 Update RMS 通常小于 1，差异来自动量，而不主要来自 $\epsilon$。

### 6.4 两个极限与一般式

- $\beta_1=0$ 时 $\kappa=1$，$\mathbb E[u_t^2]\approx1$，退化为 SignSGD/RMSProp 的行为。
- $\beta_1\to1$ 时 $\kappa\to0$，有

$$
\mathbb E[u_t^2]
\approx
\frac{g_t^2}{g_t^2+\sigma_t^2/B}
\approx(\mathbb E[u_t])^2.
$$

一般地，把 $g_t^2/(g_t^2+\sigma_t^2/B)$ 用 $\rho_B^2$ 近似，可写成

$$
\mathbb E[u_tu_t^\top]
\approx
\mu\mu^\top+\kappa\operatorname{diag}(1-\mu^2),
\qquad
\mu\approx\rho_Bs.
$$

用前面的 $S_1,D,A$ 记号，Adam 的最优学习率为

$$
\boxed{
\eta_B^*
\approx
\frac{S_1}
{\kappa D/\rho_B+\rho_B(A-\kappa D)}
}.
$$

### 6.5 对角 Hessian 下的 Adam Surge 阈值

若 $H$ 为对角阵，则 $A=D$，于是

$$
\eta_B^*
\approx
\frac{S_1}
{D\left(\kappa/\rho_B+(1-\kappa)\rho_B\right)}.
$$

分母关于 $\rho$ 的最小点是

$$
\rho^*=\sqrt{\frac{\kappa}{1-\kappa}}
=\sqrt{\frac{1-\beta_1}{2\beta_1}}.
$$

它落入 (0,1) 的条件为

$$
\beta_1>\frac13.
$$

因此，当 $\beta_1>1/3$ 时，Adam 即使在对角 Hessian 假设下也可能先增后减。由 $\rho_B^2=1/(1+\mathcal B_{\mathrm{simple}}/B)$，临界 Batch 为

$$
\boxed{
B_{\mathrm{crit}}
=\frac{1-\beta_1}{3\beta_1-1}
\mathcal B_{\mathrm{simple}}
\qquad\left(\beta_1>\frac13\right).
}
$$

这说明增大 $\beta_1$ 有两个效果：一方面通过 $B_{\mathrm{eff}}$ 降低噪声，另一方面会把系统更快推向可能发生 Surge 的大 Batch 区域。

## 7. 11593：为什么 $\beta_1=\beta_2$ 有稳定性意义

本节对应原文 11593。[^k11593]

这篇文章讨论的“最优”主要是稳定性、尺度不变性和有界更新意义下的最优，不是对所有数据集和任务都保证验证集最优。

### 7.1 相同 EMA 权重就是在线一阶、二阶矩估计

当 $\beta_1=\beta_2=\beta$ 时，bias-corrected 统计量都是同一组权重的加权平均：

$$
a_k=\frac{1-\beta}{1-\beta^t}\beta^{t-k},
\qquad
\sum_{k=1}^t a_k=1,
$$

$$
\hat m_t=\sum_{k=1}^t a_kg_k,
\qquad
\hat v_t=\sum_{k=1}^t a_kg_k^2.
$$

因此，逐坐标都有加权方差恒等式：

$$
\begin{aligned}
\hat v_t
&=\sum_k a_k\left(\hat m_t+(g_k-\hat m_t)\right)^2\\
&=\hat m_t^2+\sum_k a_k(g_k-\hat m_t)^2.
\end{aligned}
$$

交叉项为零是因为 $\sum_k a_k(g_k-\hat m_t)=0$。定义

$$
\hat\sigma_t^2=\sum_k a_k(g_k-\hat m_t)^2\ge0,
$$

就有

$$
\boxed{\hat v_t=\hat m_t^2+\hat\sigma_t^2}.
$$

忽略 $\epsilon$ 时，Adam 更新量可写成

$$
u_t
=\frac{\hat m_t}{\sqrt{\hat v_t}}
=\frac{\operatorname{sign}(\hat m_t)}
{\sqrt{1+\hat\sigma_t^2/\hat m_t^2}}.
$$

所以每个坐标满足 $\lvert u_t\rvert\le1$，且梯度波动越大，更新幅度越小。这就是“信噪比感知的 $\ell_\infty$ 最速下降”：方向接近 $\operatorname{sign}(\hat m_t)$，幅度由局部信噪比自适应决定。

### 7.2 一阶展开为什么偏向 $\beta_2=\beta_1$

设 $p_k$ 是 $\hat m_t$ 使用的归一化 $\beta_1$ 权重，$q_k$ 是 $\hat v_t$ 使用的归一化 $\beta_2$ 权重：

$$
\hat m_t=\sum_kp_kg_k,
\qquad
\hat v_t=\sum_kq_kg_k^2.
$$

把每个 $g_k$ 写为 $\hat m_t+(g_k-\hat m_t)$，得到精确分解

$$
\hat v_t
=\hat m_t^2
+2\hat m_t\left(\sum_kq_kg_k-\hat m_t\right)
+\sum_kq_k(g_k-\hat m_t)^2.
$$

如果把偏差 $g_k-\hat m_t$ 视为小量，一阶项是

$$
2\hat m_t\left(\sum_kq_kg_k-\sum_kp_kg_k\right).
$$

要使这一项对任意梯度序列都消失，必须让 $q_k=p_k$，对应

$$
\boxed{\beta_2=\beta_1}.
$$

动机是让 $\hat v_t$ 尽量贴近 $\hat m_t^2$，从而让更新接近有界的符号更新。这个一阶展开本身是近似，但 $\beta_1=\beta_2$ 时的加权方差恒等式是精确的。

### 7.3 双重优化：从最坏梯度序列直接得到相同结论

固定 $\hat m_t=m$，考虑逐坐标问题

$$
\min_{g_1,\ldots,g_t}\sum_kq_kg_k^2
\quad\text{s.t.}\quad
\sum_kp_kg_k=m.
$$

由 Cauchy-Schwarz 不等式：

$$
\left(\sum_kp_kg_k\right)^2
\le
\left(\sum_k\frac{p_k^2}{q_k}\right)
\left(\sum_kq_kg_k^2\right).
$$

所以内层最小值为

$$
\min\sum_kq_kg_k^2
=\frac{m^2}{\sum_kp_k^2/q_k},
$$

等号可由 $g_k$ 与 $p_k/q_k$ 成比例的序列取得。再对分母使用一次 Cauchy-Schwarz：

$$
\sum_k\frac{p_k^2}{q_k}
\ge
\frac{(\sum_kp_k)^2}{\sum_kq_k}=1.
$$

因此任何 $q$ 的最坏情形下界都不超过 $m^2$；当且仅当 $q_k=p_k$ 时达到这个最大下界。于是

$$
\boxed{\beta_2=\beta_1}
$$

是“让二阶统计量在任意梯度序列下尽量不小”的最坏情形稳定性选择，并且此时

$$
\hat v_t\ge\hat m_t^2,
\qquad
\lvert u_t\rvert\le1.
$$

注意：这是一个稳定性准则，不是对泛化、收敛速度或最终 Loss 的普适最优性证明。

### 7.4 为什么传统经验常用 $\beta_2>\beta_1$

$\beta_2>\beta_1$ 与 $\beta_1=\beta_2$ 针对的目标不同：

- 从 Hessian 估计角度，二阶矩希望使用更长的 EMA，以得到较稳定的梯度平方统计；一阶动量则要更局部，否则接近最优点时容易被平均到零。
- 在小模型、小 Batch、高噪声场景，更大的 $\beta_2$ 还能让分母变化更慢，使邻近更新更容易叠加，表现得更像降噪后的 SGD。
- 在大模型、大 Batch、强调长期稳定训练的场景，$\beta_1=\beta_2$ 的有界更新和尺度鲁棒性可能更重要。

因此这不是逻辑矛盾，而是不同目标、噪声规模和训练阶段下的取舍。

## 8. 10588：从 Hessian 近似理解 Adam

本节对应原文 10588。[^k10588]

### 8.1 Newton 法视角

局部二阶模型为

$$
L(\theta)
\approx L(\theta_t)+g_t^\top(\theta-\theta_t)
+\frac12(\theta-\theta_t)^\top H_t(\theta-\theta_t).
$$

若 $H_t$ 正定，最小点满足

$$
\theta_{t+1}=\theta_t-H_t^{-1}g_t.
$$

SGD 可以看成假设

$$
H_t\approx\eta_t^{-1}I.
$$

Adam 的更新是

$$
\theta_{t+1}
=\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
$$

因此若把 $\hat m_t$ 视作梯度估计，Adam 相当于使用对角的有效 Hessian

$$
H_t^{\mathrm{Adam}}
\approx
\eta_t^{-1}\operatorname{diag}(\sqrt{\hat v_t}+\epsilon).
$$

### 8.2 终点附近的长期梯度平方平均

设 $\theta^*$ 是局部最优点，$g(\theta^*)=0$。在 $\theta^*$ 附近一阶展开：

$$
g_\theta\approx H_*(\theta-\theta^*).
$$

若训练轨迹在最优点附近绕行，并近似满足

$$
\theta-\theta^*\sim\mathcal N(0,\sigma^2I),
$$

则

$$
\begin{aligned}
\mathbb E[g_\theta g_\theta^\top]
&\approx
H_*\mathbb E[(\theta-\theta^*)(\theta-\theta^*)^\top]H_*^\top\\
&=\sigma^2H_*H_*^\top.
\end{aligned}
$$

若 $H_*$ 对称且近似对角，则

$$
\operatorname{diag}(\mathbb E[g_\theta\odot g_\theta])
\approx\sigma^2H_*^2,
$$

从而

$$
\boxed{
H_*
\approx
\frac1\sigma
\operatorname{diag}\left(
\sqrt{\mathbb E[g_\theta\odot g_\theta]}
\right)
}.
$$

Adam 的 $\hat v_t$ 正是梯度平方的滑动平均，因此在这些假设下

$$
\eta_t^{-1}\operatorname{diag}(\sqrt{\hat v_t})
$$

可以看成局部对角 Hessian 的估计，差一个整体尺度 $1/(\eta_t\sigma)$。

这个推导也解释了为什么单纯从“估计 Hessian”出发会偏向较大的 $\beta_2$：二阶矩需要更长的时间窗口，而一阶动量不宜过长。

### 8.3 与 Gauss-Newton 近似的差异

对平方误差损失

$$
L(\theta)=\frac12\mathbb E\lVert y-f_\theta(x)\rVert^2,
$$

在当前点线性化模型，得到 Gauss-Newton 形式

$$
H_\theta\approx\mathbb E[J_\theta J_\theta^\top].
$$

梯度为

$$
g_\theta=\mathbb E[J_\theta r_\theta],
\qquad
r_\theta=y-f_\theta(x).
$$

若把“期望的乘积”用“乘积的期望”替换，并假设残差近似各向同性 $r_\theta\sim\mathcal N(0,\sigma^2I)$，则得到一种瞬时平均场关系：

$$
g_\theta g_\theta^\top
\approx
\sigma^2\mathbb E[J_\theta J_\theta^\top]
\approx\sigma^2H_\theta.
$$

它与前面终点附近的长期关系

$$
\mathbb E[g_\theta g_\theta^\top]
\approx\sigma^2H_*^2
$$

表面上差一个平方。两者并不必然矛盾：

- $gg^\top\approx\sigma^2H$ 是当前点、残差和 Jacobian 的瞬时平均场近似；
- $\mathbb E[gg^\top]\approx\sigma^2H^2$ 是在最优点附近对参数偏移做长期平均后的关系。

它们交换了不同的期望和乘积，并使用了不同的统计假设，不能在不加条件的情况下当作同一个 Hessian 恒等式。

### 8.4 任意参考点与 AdaBelief

若展开点 $\theta^*$ 不要求是梯度为零的最优点，则

$$
\mathbb E[(g_\theta-g_{\theta^*})(g_\theta-g_{\theta^*})^\top]
\approx\sigma^2H_{\theta^*}H_{\theta^*}^\top.
$$

这提示：如果滑动平均的不是 $g^2$，而是中心化波动 $(g-m)^2$，就能更直接地估计局部变化；AdaBelief 的二阶统计正是这一方向的实现。

## 9. 六篇文章合起来的统一结论

### 9.1 Batch Size 首先控制的是噪声，不是一个固定的线性缩放因子

在二阶局部模型中，$B$ 通过 $\Sigma/B$ 进入更新二阶矩。对 SGD，结果是

$$
\eta_B^*=\frac{\eta_{\max}}{1+\mathcal B_{\mathrm{noise}}/B}.
$$

因此“线性律”只是小 Batch 渐近，“平方根律”是 Sign/Muon 型非线性归一化在小 Batch 下的结果，二者不是普适竞争规则。

### 9.2 动量可以看成免费的有效 Batch 放大

在独立噪声、缓变梯度和稳态 EMA 假设下：

$$
B_{\mathrm{eff}}
\approx B\frac{1+\beta_1}{1-\beta_1}.
$$

这解释了动量降低噪声的统计意义，但不代表它在动态系统中与真的增大 Batch 完全等价；动量还会改变轨迹、滞后和瞬态。

### 9.3 学习率曲线与每步效率曲线必须分开看

最优学习率是 $a_B/c_B$，平均损失下降是 $a_B^2/(2c_B)$。预条件器对前者不是齐次量，对后者在一定条件下是齐次量，所以可能出现：

- 学习率随 Batch 先增后降；
- 但每步平均损失下降仍随 Batch 单调增加；
- 总步数—总数据量仍近似服从双曲关系。

这也是 Surge 不应简单理解成“大 Batch 一定更差”的原因。

### 9.4 Muon 的主要变化是更新几何，而不是标量噪声律

$\operatorname{msign}(G)$ 把梯度矩阵的奇异值归一化为 1，改变了参数空间的方向几何。在各向同性噪声和匹配 Hessian 结构下，Batch Size 只通过同一个 $\rho_B$ 进入，所以 Muon 得到 SignSGD 式的平方根—饱和规律。

### 9.5 $\beta_1=\beta_2$ 是稳定性结论，不是无条件超参定理

等 $\beta$ 的核心好处是：同一组权重构成在线均值和二阶矩，产生精确的加权方差分解

$$
\hat v_t=\hat m_t^2+\hat\sigma_t^2,
$$

从而更新有界；Cauchy-Schwarz 的双重优化还给出最坏梯度序列下的稳定性最优。

但若目标是终点附近的 Hessian 估计或小 Batch 降噪，$\beta_2>\beta_1$ 仍有合理性。选择哪一种，应先明确优化目标和训练噪声区间。

### 9.6 Adam 是一种带统计预条件的近似 Newton 法

Adam 的 $\sqrt{v_t}$ 可以在特定局部统计假设下看成对角 Hessian 的尺度估计；但真实训练中它同时受梯度噪声、EMA 记忆、参数漂移、非凸曲率和 $\epsilon$ 影响。因此“Adam 近似 Newton”是有用的解释框架，不是完整等价关系。

## 10. 假设清单与使用建议

| 假设                            | 在推导中的作用                | 失效时的风险                            |
| ------------------------------- | ----------------------------- | --------------------------------------- |
| 小步长、局部二阶 Taylor         | 得到 $\eta^*=a/c$                | 大步长时余项改变最优点                  |
| Batch 样本近似独立              | 得到 $\operatorname{Cov}(g_B)=\Sigma/B$         | 数据相关或采样相关时噪声不按 $1/B$ 降     |
| $c_B>0$                           | 保证局部二次函数有最小值      | 非凸区域可能不能这样选步长              |
| 梯度和方差缓变                  | 把历史 EMA 统计替换为当前统计 | 早期 warmup、分布漂移时结论偏差大       |
| 不同时间 Batch 噪声独立         | 推导动量方差的几何级数        | 相关噪声会改变 $\kappa$ 因子               |
| 平均场闭合                      | 计算 Sign/SoftSign/Muon 的矩  | 强重尾、强相关时误差可能很大            |
| 分量独立或矩阵各向同性          | 简化二阶矩和 $G^\top G$            | 坐标相关、谱高度不均匀时 Surge 阈值改变 |
| 特定 Hessian 结构               | 得到 Sign/Muon 的简洁 $\eta(B)$  | 一般 Hessian 下可能出现非单调性         |
| 忽略 $\epsilon$ 和 bias correction | 得到 Adam 的稳态简式          | 小梯度、训练早期和低 SNR 区域尤其不准   |

实际使用这些结果时，建议把 $\mathcal B_{\mathrm{noise}}$ 或 $\mathcal B_{\mathrm{simple}}$ 当作随训练状态变化的诊断量，而不是一次估计后永久固定的超参数。至少应区分：

1. 你要拟合的是最优学习率、每步下降量，还是达到目标所需的总数据量；
2. 你处在小 Batch 噪声区、过渡区，还是接近饱和的大 Batch 区；
3. 观测到的非单调性来自 Hessian—优化器不匹配，还是来自 EMA 瞬态、$\epsilon$、权重衰减或数据分布漂移。

## 11. 一句话版结论

在共同的二阶局部框架下，Batch Size 通过梯度噪声进入更新的二阶矩；SGD 给出“线性到饱和”，SignSGD/Muon 给出“小 Batch 平方根到饱和”，动量把 Batch 的有效噪声规模放大约 $\frac{1+\beta_1}{1-\beta_1}$ 倍，Adam 的双重 EMA 又改变了更新 RMS 并可能提前触发 Surge；$\beta_1=\beta_2$ 的主要理论理由是同权重均值—方差分解带来的有界和稳定更新，而 Adam 的 $\sqrt{v}$ 则可在局部统计假设下理解为对角 Hessian 的近似。

### 参考文献

[^k11260]: 科学空间. 《重新思考学习率与 Batch Size（一）：现状》. 2025. [原文](https://kexue.fm/archives/11260)

[^k11280]: 科学空间. 《重新思考学习率与 Batch Size（二）：平均场》. 2025. [原文](https://kexue.fm/archives/11280)

[^k11285]: 科学空间. 《重新思考学习率与 Batch Size（三）：Muon》. 2025. [原文](https://kexue.fm/archives/11285)

[^k11301]: 科学空间. 《重新思考学习率与 Batch Size（四）：EMA》. 2025. [原文](https://kexue.fm/archives/11301)

[^k11593]: 科学空间. 《Adam 优化器的最优超参数是 $\beta_1=\beta_2$？》. 2026. [原文](https://kexue.fm/archives/11593)

[^k10588]: 科学空间. 《从 Hessian 近似看自适应学习率优化器》. 2024. [原文](https://kexue.fm/archives/10588)
