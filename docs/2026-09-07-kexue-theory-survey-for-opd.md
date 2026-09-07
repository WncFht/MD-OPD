# kexue.fm 文章理论调研：OPD 谱吸收模型的映射、详细推导与理论拓展（工作底稿 v2）

**日期**：2026-09-07（v2：总库移至附录 A 并改为条目式；正文扩写前置知识与完整推导；引用全部改脚注制）
**定位**：对已拉取的 kexue.fm（科学空间）AI 板块 509 篇文章做理论清点与消化，回答三个问题：

1. 这些文章里有哪些定理/推导/恒等式（**附录 A** 理论资产总库）；
2. 最重要的结果如何搬进 OPD 谱吸收模型——给出**前置知识 + 完整推导**，每节末以「OPD 视角：改什么、拿到什么」收束（改动规格 $+$ 命题 2.x.n）（§1–§2）；
3. 拼起来之后我们自己的理论能往哪里扩（§3 拓展命题 E1–E10）。

**上游文档**：[2026-09-07-opd-efficiency-theory-brief.md](2026-09-07-opd-efficiency-theory-brief.md)（谱吸收模型最小摘要）与同目录四篇源头文档（完整推导、H1–H7、M1–M6、P1–P6）。本文不重复实验程序，只在 §4 用几行话衔接。
**方法**：509 篇按主题分 15 批并行精读提取（覆盖清单见附录 B），逐篇提取定理/恒等式/推导链/假设；公式忠实原文，OPD 映射经本轮综合校对并补全了推导。
**引用格式**：正文在作者-年份或论文名后紧跟脚注标记（小写 ASCII 键）；全部文献列于文末「参考文献」，编号由渲染器按首次引用自动生成。

---

## 0. 记号与问题设定

**OPD 目标**（Fu et al. 2026[^fu26]；系列首篇见[^opd1-26]；EffOPD 见 Cai et al. 2026[^cai26]）：冻结一批学生自己采样的状态集 $\mathcal S$ 后，最小化固定经验目标

$$J(\theta)=\mathbb E_{s\in\mathcal S}\,\mathrm{KL}\big(\pi_\theta(\cdot\mid s)\,\big\|\,\pi_T(\cdot\mid s)\big).$$

这是**学生采样状态上的 reverse-KL**（学生分布作第一分量），与 DAgger（Ross et al. 2011[^ross11]）、MiniLLM（Gu et al. 2024[^gu24]）、GKD（Agarwal et al. 2024[^agarwal24]）同源但目标固定；梯度用逐 token advantage $A_i=\log\pi_T(y_i\mid s_i)-\log\pi_\theta(y_i\mid s_i)$ 估计（教师项不反传；完整梯度含 advantage 后缀和，实践用 immediate-token 近似）。

**要解释的四个经验事实**：吸收率 $v_t=(d_t-d_{t+1})/d_t$ 单调下降；节奏与数据规模无关；学习率只重标时间轴（lr 折叠 $u=\eta t$）；冻结状态集后仍稳定对齐约 200 步才进入平台。四条合起来把解释对象钉死为**固定目标上的参数空间优化动力学**。

**记号表**（与源头文档一致，本文 §1 重推其来历）：

| 记号                 | 含义                                                           |
| -------------------- | -------------------------------------------------------------- |
| $\lambda_i,\ e_i$    | 局部二次化后 Hessian（$=$Fisher 型）的特征值/特征方向          |
| $c_i$                | 初始误差在模式 $i$ 上的幅度（$x_i(0)=c_i$）                    |
| $\sigma_i^2,\ B$     | 模式 $i$ 的单样本梯度噪声方差；batch 大小                      |
| $d(t)$               | 距离读数 $=\sum_i a_ie^{-\eta\lambda_it}$（谱的 Laplace 变换） |
| $v_t,\ \alpha(t)$    | 吸收率；$\alpha(t)=-\frac{d\ln d}{d\ln t}=t\,v_t$              |
| $\lambda_{\rm crit}$ | 可吸收阈值 $\frac{\sigma_i}{c_i}\sqrt{\eta/(2B)}$              |
| $d_\infty$           | 终态底 $=\frac{\eta}{4B}\sum_i\sigma_i^2+d_{\rm gap}$          |

**三旋钮语法**：任何优化干预 $=$ 修改映射 $\{\lambda_i,c_i,\sigma_i\}\to\{\eta\lambda_i^{\rm eff},\ \lambda_{\rm crit},\ d_\infty\}$；三个旋钮互不替代。

---

## 1. 前置知识（工具箱）

### 1.1 token 分布上的 KL：Hessian=Fisher 的精确陈述与残差项

这是整个模型的曲率输入（源头文档前提 C2/A2），值得把"精确"与"近似"的边界推清楚。单个状态 $s$ 上，记教师分布 $p=\pi_T(\cdot\mid s)$、学生 $q=q(z)=\pi_\theta(\cdot\mid s)$，$z$ 为该位置的 logits，$s_y=\log q_y$，$t_y=\log p_y$，逐 token advantage $A_y=t_y-s_y$（$=$ OPD 的 $\log\pi_T-\log\pi_\theta$）。softmax 的基本恒等式（下文反复使用）：

$$
\frac{\partial q_y}{\partial z_k}=q_y(\delta_{yk}-q_k),\qquad
\frac{\partial s_y}{\partial z_k}=\delta_{yk}-q_k,\qquad
\sum_y q_y\frac{\partial s_y}{\partial z_k}=0.
$$

**(a) forward-KL 的 Hessian 精确等于模型 Fisher。** 取 $L(z)=\mathrm{KL}(p\|q)=\mathrm{const}-\sum_y p_ys_y$：

$$
\frac{\partial L}{\partial z_k}=-\sum_yp_y(\delta_{yk}-q_k)=q_k-p_k,\qquad
\frac{\partial^2L}{\partial z_k\partial z_l}=\frac{\partial q_k}{\partial z_l}=q_k(\delta_{kl}-q_l).
$$

而模型自身的（真）Fisher 是

$$
F_q=\mathbb E_{y\sim q}\big[\nabla_z\log q_y\,\nabla_z\log q_y^{\top}\big]
=\mathbb E_{y\sim q}\big[(e_y-q)(e_y-q)^{\top}\big]=\mathrm{diag}(q)-qq^{\top}.
$$

所以 $\nabla_z^2\mathrm{KL}(p\|q)=F_q$ **对任意参照分布 $p$ 精确成立**——forward-KL 是 $\log Z$ 型 Bregman 散度，曲率与参照无关。这是"softmax 下 GGN$=$Fisher 精确"的标准出处；真 Fisher 与 empirical Fisher 在解附近并不重合（Kunstner et al. 2019[^kunstner19]），源头文档 A3 采用真 Fisher 口径即基于此。

**(b) reverse-KL：梯度是中心化 advantage 的策略梯度，Hessian $=F_q+$ 优势色散修正。** 取 $R(z)=\mathrm{KL}(q\|p)=\sum_yq_ys_y-\sum_yq_yt_y=-H(q)-\mathbb E_q[t]$。记中心化优势 $\bar u_y:=A_y-\langle A\rangle_q$（$\langle\cdot\rangle_q$ 为 $q$ 期望）。利用 $\sum_yq_y\,\partial s_y/\partial z_k=0$ 与 $\partial\langle A\rangle_q/\partial z_k=q_k\bar u_k$（对 $\sum_y\frac{\partial q_y}{\partial z_k}A_y$ 直接展开）：

$$
\frac{\partial R}{\partial z_l}
=\sum_y\frac{\partial q_y}{\partial z_l}\big(s_y-t_y+1\big)
=-q_l\big(A_l-\langle A\rangle_q\big)=-q_l\bar u_l.
$$

这正是**中心化 advantage 的 REINFORCE 梯度**——OPD 实际使用的形式；中心化基线不改变期望、只降方差（Greensmith et al. 2004[^greensmith04]；最优 baseline 的显式形式见[^k7737]）。再求一次导（$\partial s_l/\partial z_m=\delta_{lm}-q_m$，$\partial\langle s\rangle_q/\partial z_m=0$）：

$$
\frac{\partial^2R}{\partial z_m\partial z_l}
=-\frac{\partial q_l}{\partial z_m}\bar u_l-q_l\frac{\partial\bar u_l}{\partial z_m}
=q_l(\delta_{lm}-q_m)(1-\bar u_l)+q_lq_m\bar u_m,
$$

（其中 $\partial\bar u_l/\partial z_m=-(\delta_{lm}-q_m)-q_m\bar u_m$；对称性 $\partial^2R/\partial z_m\partial z_l=\partial^2R/\partial z_l\partial z_m$ 可直接验证：对 $l\ne m$，两项之差化简为 $q_lq_m[(\bar u_l-\bar u_m)+(\bar u_m-\bar u_l)]=0$。）写成二次型最有用。对任意扰动方向 $v$，利用 $\langle\bar u\rangle_q=0$ 与 $R$ 对 logits 平移不变（$H\cdot\mathbf 1=0$），在对移不变子空间（$\langle v\rangle_q=0$）上：

$$\boxed{\ v^{\top}\Big(\nabla_z^2R\Big)v\ =\ v^{\top}F_qv\ -\ \mathrm{Cov}_q\big(\bar u,\ v^2\big).\ }$$

**读法**：reverse-KL 的曲率 $=$ 模型 Fisher $-$ 一个由**优势色散**（$\bar u$ 与 $v^2$ 的协方差，三阶矩型量）构成的修正。当 $q\to p$ 时 $A$ 退化为常数、$\bar u=0$，修正消失、$H=F_q$ 精确；优势分布越宽（师生分歧越大），修正越可观且可正可负。这把源头文档 A2 的"失配残差项 $R$"从假设变成显式泛函：**残差由逐 token advantage 分布的形状决定**，而 advantage 的各阶矩恰是逐 token 日志可测的量；它与 11814 的 softmax 高阶泰勒展开（附录 A §A.6）是同一件事的两侧——logit 空间的三/四阶中心矩。

**(c) 参数空间。** 链式法则把 logits 曲率搬回参数空间：$H_\theta\approx J_z^{\top}F_qJ_z$（GGN），$J_z=\partial z/\partial\theta$——"局部二次化后 $H=$Fisher"的准确含义。

### 1.2 谱吸收模型：五步最小推导

以下五步是全文的公共前置，全部在固定目标（A1 冻结状态集）下进行。

**第一步（二次化）**。由 1.1，$J(\theta_0+\delta)=J(\theta_0)+\tfrac12\delta^{\top}H\delta+O(\Vert\delta\Vert^3)$，$H=\sum_i\lambda_ie_ie_i^{\top}$（Fisher 型、PSD）加残差项；残差的有效幅度由 advantage 分布的高阶矩控制（1.1(b)）。

**第二步（模式解耦）**。一阶更新带预条件 $P$：$\delta_{t+1}=(I-\eta PH)\delta_t+\eta\zeta_t$。若 $P$ 准静态（C3）且与 $H$ 近似对易，则逐模式

$$x_i(t+1)=(1-\eta\lambda_i)\,x_i(t).$$

动量/阻尼把一阶递推升级为二阶（E5），Adam 使 $\eta\lambda_i^{\rm eff}$ 逐坐标化并垫软底（§2.6）——这些是对第二步的畸变；模型主干先按纯一阶建立。

**第三步（Laplace 表示）**。解递推、代入二次型距离：$d(t)=\tfrac12\sum_i\lambda_ix_i^2(t)=\sum_i a_ie^{-\eta\lambda_it}$，$a_i=\tfrac12\lambda_ic_i^2$（KL 口径；L1 口径见 §2.7）。**对数距离曲线是初始振幅加权谱测度的 Laplace 变换**——全文最重要的恒等式，一切读数理论从这里出发。

**第四步（指数倾斜与递减节奏）**。距离权重随时间指数倾斜：

$$
w_i(t)\propto a_ie^{-\eta\lambda_it},\qquad v_t\simeq\eta\langle\lambda\rangle(t),\qquad
\frac{d\langle\lambda\rangle}{dt}=-\eta\,\mathrm{Var}_{w(t)}(\lambda)\le0,
$$

最后一步逐行：$d\ln w_i/dt=-\eta\lambda_i$，故 $\frac{d\langle\lambda\rangle}{dt}=\sum_iw_i(-\eta\lambda_i)(\lambda_i-\langle\lambda\rangle)=-\eta(\langle\lambda^2\rangle-\langle\lambda\rangle^2)$。**"$v_t$ 下降"因此是定理而非机制**：$d(t)$ 是指数混合，指数混合必有递减风险率（Proschan 1963[^proschan63]；完全单调函数的标准性质见 Feller[^feller71] 第 XIII 章）。定量形状（幂律/带边弯曲/平台）才携带谱信息——见 §2.5 与 E1。

**第五步（两个读数恒等式）**。$\alpha(t)\equiv-\frac{d\ln d}{d\ln t}=t\cdot v_t^{\rm inst}$（log-距离面板与 log-吸收率面板携带同一信息）；以及 §2.3 的终点恒等式（终点可由轨迹加权重构）。

### 1.3 噪声底、可吸收阈值与白化极限

**离散 OU 推导**（源头文档推导五；OU 过程经典结果 Uhlenbeck & Ornstein 1930[^uhlenbeck30]；SGD 近似贝叶斯后验的系统版本 Mandt et al. 2017[^mandt17]；SDE 视角 Li–Tai–E[^li17sme] 与 Smith & Le[^smith18bayes]）。每模式把更新写成漂移 $+$ 噪声：$x_{t+1}=(1-\eta\lambda_i)x_t+\eta\xi_t$，$\mathbb E[\xi_t^2]=\sigma_i^2/B$。平稳方差只需 $\mathbb E[x_{t+1}^2]=(1-\eta\lambda_i)^2\mathbb E[x_t^2]+\eta^2\sigma_i^2/B$ 的不动点：

$$\mathbb E[x_i^2]_\infty=\frac{\eta^2\sigma_i^2/B}{1-(1-\eta\lambda_i)^2}\approx\frac{\eta\,\sigma_i^2}{2\lambda_iB}\qquad(\eta\lambda_i\ll1).$$

**推论一（KL 底逐模式与 $\lambda_i$ 无关）**：每模式对距离贡献 $\tfrac12\lambda_i\mathbb E[x_i^2]_\infty=\frac{\eta\sigma_i^2}{4B}$——曲率在分子分母恰好消掉，求和得 $d_\infty^{\rm KL}=\frac{\eta}{4B}\sum_i\sigma_i^2+d_{\rm gap}$。**推论二（L1 口径不同）**：$\mathbb E|x_i|_\infty\propto\sqrt{\eta\sigma_i^2/(\lambda_iB)}$——平坦模式主导、标度 $\propto\sqrt{\eta/B}$（§2.7 统一为"读数泛函的阶"）。

**可吸收阈值**：模式可被吸收要求初始幅度压过自身噪声底，$c_i^2\gg\frac{\eta\sigma_i^2}{2\lambda_iB}$。这个判据在"以 $u=\eta t$ 为时间单位"的读法下等价于**单位 $u$ 时间内漂移对噪声的胜利**：换元后每单位 $u$ 的漂移为 $\lambda_ic_i$、噪声标准差为 $\sqrt\eta\,\sigma_i/\sqrt B$，于是

$$
\sqrt B\,\frac{c_i\lambda_i}{\sigma_i}\ \gg\ \sqrt\eta
\quad\Longleftrightarrow\quad
\lambda_i\ \gg\ \lambda_{\rm crit}\ :=\ \frac{\sigma_i}{c_i}\sqrt{\frac{\eta}{2B}}\quad(\text{差 }\sqrt2\text{ 因子取决于窗口约定}).
$$

> **口径注记**：若把判据严格读作"初始幅度 vs OU 均衡方差"（$c_i^2\gg\mathbb E[x_i^2]_\infty$），整理后是 $\lambda_i\gg\frac{\eta\sigma_i^2}{2Bc_i^2}$——$B,c_i,\sigma_i$ 的标度相同、但 $\eta$ 是一次幂而非平方根幂。两个读法在 $\eta$ 幂次上不同，恰可由扫 $\eta$ 看 $\lambda_{\rm crit}$ 移动的实验分辨；建议在源头文档中统一约定。

**白化极限与估计论天花板**。精确自然梯度 $\theta\leftarrow\theta-\eta F^{-1}\nabla J$ 在模式坐标里是 $x_i\leftarrow(1-\eta)x_i$：所有模式同速收缩、$v\equiv\eta$ 恒定（Amari 1998[^amari98]；自然策略梯度对 softmax 策略有闭式——更新就是把中心化 advantage 加到 logits 上，Kakade 2002[^kakade02]；TRPO/PPO/Mirror-DPO 是其不同近似[^schulman15][^schulman17][^tomar21]）。但白化受信噪比限制：把 $g_i/\lambda_i$ 当更新需 $\sqrt B\,c_i\lambda_i/\sigma_i\gg1$——与 $\lambda_{\rm crit}$ 是同一阈值；$\lambda_i\to0$ 处逆预条件方差爆炸，唯一补救是 damping，且"没有单一固定 damping 适合训练全程"（Martens 2014[^martens14]）。随机凸优化的信息论下界（$O(1/t)$ 型）说明这不是实现缺陷：要往底以下走必须降方差（Agarwal et al. 2012[^agarwal12]；Bach & Moulines 2013[^bach-moulines-13]；迭代平均的系统分析见 Dieuleveut, Flammarion & Bach 2017[^dieuleveut-17]）。

### 1.4 四件优化理论工具

**T1 距离恒等式与 regret 界**。对任意参考点 $\varphi$ 与步长序列 $\{\eta_t\}$：

$$\Vert\theta_{t+1}-\varphi\Vert^2=\Vert\theta_t-\varphi\Vert^2-2\eta_t\langle g_t,\theta_t-\varphi\rangle+\eta_t^2\Vert g_t\Vert^2,$$

凸性把内积换成损失差；除以 $2\eta_t$ 求和、telescoping，得（有界域版）

$$\frac1T\sum_tE[L(\theta_t)-L(\varphi)]\le\frac{R^2}{2T\eta_T}+\frac{G^2}{2T}\sum_t\eta_t$$

——**右端 $=$ 瞬态项 $+\propto\eta$ 的底**，是 $d(t)=$瞬态$+d_\infty(\eta)$ 的凸版原型（Zinkevich 2003[^zinkevich03] 的 online convex programming 框架；无界域版与收敛条件 $\sum\eta_t=\infty,\ \sum\eta_t^2/\sum\eta_t\to0$ 见「炼丹一/二」[^k9902][^k11469]）。注意结构：**底 $\propto\eta$ 来自 $\eta_t^2/\eta_t=\eta_t$**——与 §1.3 的 OU 底同源。

**T2 变分法求最优调度**。把 T1 的界对 $\{\eta_t\}$ 极小化：令剩余步长和 $S(t)=\sum_{s\ge t}\eta_s$，界的连续化为 $c_1/S(0)+c_2\int_0^T\dot S^2/S\,\mathrm dt$，Euler–Lagrange 给 $\dot S^2\propto S$，解 $S(t)\propto(T-t)^2$，即 $\eta_t=-\dot S\propto(T-t)$：**线性衰减**；常数 $\propto1/\sqrt T$ 时达信息论下界（Defazio et al. 2023[^defazio23]；「炼丹四」[^k11494]）。

**T3 Abelian/Tauberian 定理（谱尾 $\leftrightarrow$ 大 $t$ 渐近）**。若谱密度在 $\lambda\to0$ 处 $\rho(\lambda)\propto\lambda^{s-1}$，则其 Laplace 变换 $d(t)=\int e^{-\eta t\lambda}\rho(\lambda)\,\mathrm d\lambda$ 满足 $d(t)\propto\Gamma(s)(\eta t)^{-s}$（Karamata 型 Tauberian 定理，Feller[^feller71] XIII.5）。**这是 E1 的数学核心：实测 $\alpha$ 直接反演谱尾指数 $s=\alpha$，无需数值反演。**

**T4 递减风险率与 stretched-exponential 对照**。指数混合 $\int e^{-\eta t\lambda}\,\mathrm d\mu(\lambda)$ 的风险率 $-d\ln d/dt$ 递减（Proschan[^proschan63]）；若谱是对数正态/分层约束混合，大 $t$ 行为改为 stretched exponential（Palmer et al. 1984[^palmer84]）。**"纯幂律 vs 带边弯曲"的形状判别因此有明确谱含义：幂律谱给幂律，分层/对数正态谱给弯曲。**

---

## 2. 核心移植：八个结果的完整推导

### 2.1 动量与权重平均的方差几何（EMA/动量 $=$ 等效 batch 放大）

**前置**：动量的两种经典诠释——蛙跳离散的阻尼二阶 ODE（Sutskever et al. 2013[^sutskever13]；可视化解说 Goh 2017[^goh17]）与梯度低通滤波。「炼丹七」[^k11804] 证明权重平均与步长调度的等价；SWA/Polyak 是其实证面（Polyak & Juditsky 1992[^polyak92]；Izmailov et al. 2018[^izmailov18]）；Schedule-Free 把平均嵌进训练点选择（Defazio et al. 2024[^defazio24]）。

**推导（动量的噪声压缩因子）**。设 $g_t=g+\xi_t$，$\xi_t$ 零均值、协方差 $\Sigma/B$。动量 $m_t=\beta m_{t-1}+(1-\beta)g_t$ 展开为 $m_t=(1-\beta)\sum_{k\ge0}\beta^kg_{t-k}$，于是

$$
\mathbb E[m_t]=g,\qquad
\mathrm{Cov}[m_t]=(1-\beta)^2\sum_{k\ge0}\beta^{2k}\cdot\frac{\Sigma}{B}
=\frac{(1-\beta)^2}{1-\beta^2}\cdot\frac{\Sigma}{B}
=\frac{1-\beta}{1+\beta}\cdot\frac{\Sigma}{B}.
$$

即梯度噪声方差乘 $\kappa:=\frac{1-\beta}{1+\beta}$，**等效 batch 放大 $B_{\rm eff}=B\frac{1+\beta}{1-\beta}$**（$\beta_1=0.9$：$B_{\rm eff}=19B$）。代入 §1.3：

$$
\lambda_{\rm crit}(\beta)=\frac{\sigma_i}{c_i}\sqrt{\frac{\eta}{2B_{\rm eff}}}\ \ (\text{缩 }\sqrt{19}\approx4.4\times),\qquad
d_\infty^{\rm KL}(\beta)=\frac{\eta}{4B_{\rm eff}}\sum_i\sigma_i^2+d_{\rm gap}\ \ (\text{缩 }19\times),
$$

而谱形状（$\eta\lambda_i$ 谱）与 lr 折叠不变——**动量/EMA 只动旋钮 2/3，不动旋钮 1**。「重新思考 lr 与 BS（四）」[^k11301] 用 $m_t$ 方差平方可加独立得到同一因子，并指出 Adam 的稳态二阶矩 $\mathbb E[\tilde\varphi_B^2]\approx\frac{g^2+\kappa\sigma^2/B}{g^2+\sigma^2/B}$ 分子分母同受 $B$ 影响（低 SNR 模式打折，见 §2.6）。

**权重平均 $=$ 免费的终点退火**。「炼丹七」的双权恒等式：对任意两组权重 $w,v$（对应步长 $\eta_t=\frac{v_{1:T}}{v_{t+1:T}}\frac{w_tw_{t+1:T}}{w_{1:T}}$）

$$\mathbb E\Big[L\Big(\frac1{v_{1:T}}\sum_tv_t\theta_t\Big)-L(\theta^*)\Big]\le\frac1{2w_{1:T}}\Big(R^2+\sum_tw_t^2G_t^2\Big),$$

即**以权 $v$ 平均的终点效果等价于把调度乘 $v_{t+1:T}/v_{1:T}$（单调降 $=$ Decay）**；取 $v_t=\gamma^{-t}$（EMA）得平均窗口 $\mathcal O((1-\gamma)^{-1})$：**EMA 在不改 $\eta$、不改吸收速率、不破坏 lr 折叠的前提下拿到衰减调度的终态**——"方差工程不牺牲节奏"的定理化。⚠️ 配套约束（「AdamW Weight RMS 渐近」[^k11307][^k11404]）：终点 lr$\to0$ 使权重 RMS $\rho\approx\sqrt{\eta/2\lambda_{\rm wd}}$ 坍缩（动态方程 $\frac{d\rho}{dt}\approx-2\lambda_{\rm wd}\eta\rho+\eta^2$），退火须配 weight decay $\lambda_{\rm wd}\propto\eta$。

**Adam $+$ 大 batch 的 Surge 警告**：$\mathbb E[\tilde\varphi_B^2]$ 的分母随 $B$ 增大向 $g^2$ 集中，最优 lr 随 $B$ 可先升后"涌"（Li et al. 2024[^li24surge]；平均场推导与 erf 近似见「重新思考 lr 与 BS（二）」[^k11280]）。对 OPD 的含义：lr 折叠 $u=\eta t$ 在大 $B$ 端可能破缺，折叠残差 $\sim10\%$ 的一部分可归于此。

**OPD 视角：改什么、拿到什么**

*改动（落到训练器与日志的具体位置）：*

1. **把 $\beta_1$ 从超参升级为方差预算旋钮**。OPD 的 Adam 配置按 $(\eta,B_{\rm eff})$ 而非 $(\eta,B)$ 书写：底预算确定后由 $B_{\rm eff}=B\frac{1+\beta_1}{1-\beta_1}$ 反解 $\beta_1=\frac{B_{\rm eff}-B}{B_{\rm eff}+B}$——显存与真实 batch 都不变，等效 batch 是免费的。
2. **出口换权重平均**：取 EMA 系数 $\bar\gamma$ 使窗口 $(1-\bar\gamma)^{-1}\sim$ 平台时标；weight decay 与 $\eta$ 锁比（$\lambda_{\rm wd}=c\,\eta$），避免退火期权重漂移（[^k11307][^k11404]）。
3. **日志双轨**：同一 run 对 raw $\theta$ 与 EMA $\theta$ 各记一份 $d(t),v_t$——两者的差本身是读数（命题 2.1.3）。

*理论收益：*

- **命题 2.1.1（底的动量缩减，过阻尼区）**：$d_\infty(\beta_1)=\kappa\,d_\infty(0)$，$\lambda_{\rm crit}$ 同比缩 $\sqrt{(1+\beta_1)/(1-\beta_1)}$，谱形状与 lr 折叠不变（推导：本节 $\mathrm{Cov}[m_t]$ 代入 §1.3 的 OU 平衡；准静态近似下成立，欠阻尼修正见 E5）。**反向用法**：若实测底降幅 $\ll\kappa^{-1}$，则底的主导成分不是梯度噪声（指向 $d_{\rm gap}$ 或 §2.6 软底）——底成分分解的第一个判据。归属：旋钮 3（连带 2）。
- **命题 2.1.2（免费退火定理）**：EMA 终点的期望损失 $\le$ 以 $w_t$ 加权的单点界（[^k11804]），等价于调度乘 $\frac{v_{t+1:T}}{v_{1:T}}$——**在不触碰 $\eta\lambda$ 谱的前提下拿到衰减级终点**。归属：旋钮 3 的"免费份额"。
- **命题 2.1.3（权重 EMA 的模式选择性滤波）**：对定态底，权重 EMA 的缩减因子逐模式为
  $$R_i=\frac{1-\bar\gamma}{1+\bar\gamma}\Big[1+\frac{2\bar\gamma(1-a_i)}{1-\bar\gamma(1-a_i)}\Big],\qquad a_i=\eta\lambda_i$$
  （推导：AR(1) 级联的二阶矩，交叉项 $2\bar\gamma(1-\bar\gamma)\mathbb E[\bar y_{t-1}x_t]$ 不可丢）。慢模式（$\eta\lambda_i\ll1-\bar\gamma$）$R_i\to1$：**EMA 压"快噪声底"、不压"慢漂移底"**；快模式缩至 $\sim\frac{(1-\bar\gamma)(1+2\bar\gamma)}{1+\bar\gamma}$。用法：EMA/raw 底比值随模式位置的变化把 $d_\infty$ 分解为快/慢两截。归属：旋钮 3 的细分。
- **命题 2.1.4（Surge 边界）**：$\beta_1>\frac13$ 且 $B>\frac{1-\beta_1}{3\beta_1-1}\mathcal B_{\rm simple}$ 时最优 lr 随 $B$ 反转（[^k11301][^li24surge]）——OPD 大 batch 端的折叠破缺有解析边界。归属：读数（折叠律的适用域）。

### 2.2 闭环学习率：从距离恒等式到最优调度（P1 的可证明骨架）

**前置**：T1（§1.4)。「炼丹三/四/五/六」[^k11480][^k11494][^k11530][^k11540] 构成一条完整链条。

**第一步（加权终点恒等式）**。定义尾窗和 $\lambda_k=\sum_{t=T-k+1}^Tw_t$、尾窗平均 $S_k=\lambda_k^{-1}\sum_{t=T-k+1}^Tw_tq_t$。由 $\lambda_kS_k=\lambda_{k+1}S_{k+1}-w_{T-k}q_{T-k}=\lambda_kS_{k+1}+\frac{w_{T-k}}{\lambda_{k+1}}\sum_{t=T-k}^Tw_t(q_t-q_{T-k})$，两边除以 $\lambda_k$、对 $k=1\sim T-1$ 求和（用 $\frac{w_{T-k}}{\lambda_k\lambda_{k+1}}=\frac1{\lambda_k}-\frac1{\lambda_{k+1}}$ telescoping），得**对任意序列精确**的恒等式

$$q_T=\frac1{w_{1:T}}\sum_{t=1}^Tw_tq_t+\sum_{k=1}^{T-1}\Big(\frac1{w_{k+1:T}}-\frac1{w_{k:T}}\Big)\sum_{t=k}^Tw_t(q_t-q_k).$$

**第二步（配基本不等式）**。把 T1 的距离恒等式起点改到 $k$、取 $\varphi=\theta_k$：$\sum_{t=k}^T\eta_t\,\mathbb E[L(\theta_t)-L(\theta_k)]\le\frac{G^2}2\sum_{t=k}^T\eta_t^2$（**不需要 $\eta_t$ 单调**）。代入 $q_t=\mathbb E[L(\theta_t)-L(\theta^*)]$、$w_t=\eta_t$：

$$\mathbb E[L(\theta_T)-L(\theta^*)]\le\frac{\Vert\theta_1-\theta^*\Vert^2}{2\eta_{1:T}}+\frac{G^2}2\sum_t\frac{\eta_t^2}{\eta_{\min(t+1,T):T}}.$$

**第三步（变分）**。对 $\{\eta_t\}$ 极小化上界：令 $S(t)=\sum_{s\ge t}\eta_s$，右端连续化为 $\frac{c_1}{S(0)}+\frac{c_2}2\int_0^T\frac{\dot S^2}{S}dt$；Euler–Lagrange 给 $S(t)\propto(T-t)^2$，即 **$\eta_t\propto(T-t)$ 线性衰减**，常数 $\propto1/\sqrt T$ 时达下界。换用梯度信息（「炼丹五/六」[^k11530][^k11540]）得闭环版：

$$\boxed{\ \eta_t=\frac{R\,G_t^{-2}}{\sqrt{Q_T}}\Big(1-\frac{Q_t}{Q_T}\Big),\qquad Q_t=\sum_{k\le t}G_k^{-2}\ }$$

即 **Warmup（$G^{-2}$ 反比律）$\times$ 终点线性退火**；11540 用构造 $\psi_{t+1}=\psi_t-w_tg_t$（可验证 $\psi_{t+1}-\psi_t=\frac{w_{1:T}}{w_{t+1:T}}(\theta_{t+1}-\theta_t)$）**严格证明**了该律最优，且给出主项常数 $1/4$ 的显式界。

**OPD 化**。谱模型里 $G_t^2=\Vert g_t\Vert^2=\sum_i\lambda_i^2x_i^2(t)=2\,J(t)\,\langle\lambda\rangle_{J{\text{-w}}}(t)$：梯度模长同时携带剩余距离与活性曲率。而 $v_t\simeq\eta\langle\lambda\rangle(t)$ 正是活性曲率的在线读数，故可操作反馈律为

$$\eta_{t+1}\ \propto\ \frac1{G_t^2}\ \propto\ \frac1{d(t)\,\langle\lambda\rangle(t)}\ \propto\ \frac{\eta}{d(t)\,v_t}.$$

**与底的权衡**：升 lr 抬 $d_\infty\propto\eta$（T1 的 $\eta_t$ 底与 OU 底同源），所以闭环升 lr 单独不降底——(i) 中期沿 $1/(d\,v_t)$ 升，(ii) 末端按 $(1-Q_t/Q_T)$ 退火到 0（否则按线性衰减下界亏终点），(iii) 与 §2.1 的方差工程叠加压底。「炼丹八」的多阶段组合律（等效步数 $T_2^\#=\frac12T_2+\frac12\sqrt{T_1(2T_2-T_1)}$；计划式两阶段的 minimax 有球交几何闭式）进一步给出 checkpoint 续训的 lr 组合规则[^k11879]。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **反馈律落地为两段式**：中期 $\eta_{t+1}\propto\eta/(d(t)\,v_t)$（活性曲率读数驱动），末端乘 $(1-t/T)$ 线性退火（T2）；梯度模长形式 $\eta_t\propto G_t^{-2}$ 与读数形式等价（$G_t^2=2\,d\,\langle\lambda\rangle$），二选一以日志质量定。
2. **先 Refinement 后在线化**：对已有 run 的日志离线重算 $\eta_{\rm opt}(t)\propto G_t^{-2}(1-Q_t/Q_T)$ 并与实际调度对照（零风险）；一致性通过后再上在线反馈。
3. **反馈是一个干预包**：升 lr 必须同时按命题 2.1.1 配 $\beta_1$（压 $\kappa$）、$\lambda_{\rm wd}\propto\eta$（防坍缩，[^k11404]）——三件事捆绑，不是三个独立开关。

*理论收益：*

- **命题 2.2.1（P1 的界与最优律）**：OPD 二次化域内，闭环 lr 的终点界为 $\frac{R^2}{2\eta_{1:T}}+\frac{G^2}2\sum_t\frac{\eta_t^2}{\eta_{\min(t+1,T):T}}$，最优解即 $\eta_t=\frac{RG_t^{-2}}{\sqrt{Q_T}}(1-\frac{Q_t}{Q_T})$（[^k11530][^k11540]）。归属：旋钮 1（同形状曲线上的超线性前进）。
- **命题 2.2.2（P1 净收益的不等式）**：升 lr 使底 $\propto\eta$ 同步抬升，P1 的净收益 $=$ 中段加速 $-$ 底抬升；配 EMA（$\kappa$ 压底）后净收益转正的条件可写成显式不等式（底项被 $\kappa$ 压制的倍数 vs 加速倍数）——把 brief"P1 必须配方差工程"从直觉变成不等式。归属：旋钮 1×3 的耦合。
- **命题 2.2.3（折叠律的闭环推广）**：时变 $\eta_t$（缓变：$\dot\eta/\eta\ll\lambda_i$ 逐模式）下，模式解耦的乘积公式给
  $$x_i(t)=x_i(0)\prod_s(1-\eta_s\lambda_i)\ \approx\ x_i(0)\,e^{-\lambda_iU(t)},\qquad U(t)=\int_0^t\eta_s\,\mathrm ds,$$
  即 $d(t)=\sum_ia_ie^{-\lambda_iU(t)}$ 对任意缓变调度精确成立——**折叠变量从 $u=\eta t$ 推广为累计步长 $U$**；$v_t\simeq\eta_t\langle\lambda\rangle(U)$。可证伪：闭环 run 的 $d$–$U$ 曲线应与常数 lr 的 $d$–$u$ 曲线重合。归属：读数。

### 2.3 终点恒等式：$d(T)$ 的无噪重构

「炼丹三」[^k11480] 的恒等式是 §2.2 第一步取 $w\equiv1$ 的特例。直接推导：定义尾窗平均 $S_k=\frac1k\sum_{t=T-k+1}^Tq_t$（$S_T=\frac1T\sum_{t\le T}q_t$，$S_1=q_T$）。由 $S_{k+1}=\frac{kS_k+q_{T-k}}{k+1}$ 得

$$
S_{k+1}-q_{T-k}=\frac1{k+1}\sum_{t=T-k}^T(q_t-q_{T-k}),\qquad
\frac1k\,(S_{k+1}-q_{T-k})=\Big(\frac1k-\frac1{k+1}\Big)\sum_{t=T-k}^T(\cdots);
$$

对 $k$ 求和，左侧 telescoping 为 $S_1-S_T=q_T-\frac1T\sum q_t$，右侧即得

$$q_T=\frac1T\sum_{t=1}^Tq_t+\sum_{k=1}^{T-1}\frac1{k(k+1)}\sum_{t=T-k}^T\big(q_t-q_{T-k}\big).$$

**读法（OPD）**：取 $q_t=d(t)-d_\infty$。左端是终点剩余距离；右端全部由单 run 的 $d(t)$ 轨迹算出，权重 $\frac1{k(k+1)}$ 把权重集中到尾部（$k$ 小 $\Rightarrow t\approx T$）。两个用途：(i) **读数降噪**——窗口平均方差 $\propto1/$窗口，该恒等式是无偏的尾窗滤波器；(ii) **$d_\infty$ 的轨迹估计**——对候选 $d_\infty$ 要求恒等式两端一致，解出平台值而不必等收敛。「炼丹四/七」的加权版允许直接用 $v_t$ 作权（双权恒等式，[^k11494][^k11804]）。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **读数替换**：日志后处理把单点 $d(t)$ 补充为尾窗读数 $\hat q_T=\frac1T\sum_tq_t+\sum_k\frac1{k(k+1)}\sum_{t=T-k}^T(q_t-q_{T-k})$——零训练成本，纯后处理；权重可换 $v_t$（双权版）。
2. **平台估计器**：对候选 $d_\infty$ 解恒等式两端的一致性（一维求根），配合窗口方差给置信区间；无需等 run 收敛。
3. **与谱反演互锁**：恒等式给"轨迹 $\to$ 终点"，NNLS 反演给"轨迹 $\to$ 谱"；把反演谱代回恒等式做残差检验，两套估计互相约束。

*理论收益：*

- **命题 2.3.1（谱模型下的早停外推界）**：恒等式对每个模式分别成立（对 $\{a_i,\lambda_i\}$ 线性）；用 $t\le T/2$ 的窗口预测 $d(T)$，误差由窗口外谱尾的质量 $\int_0^{\lambda(T/2)}\rho(\lambda)\,\mathrm d\lambda$ 决定——**外推精度有谱判据**，且与 §2.5 的谱尾指数衔接（尾越平越难外推）。
- **命题 2.3.2（读数方差的显式缩减）**：权重 $\frac1{k(k+1)}$ 的有效窗口 $\sim$ 若干个弛豫时标，方差缩减因子 $\approx$ 有效窗口大小——把 A6"无噪读数"假设放松为"已知窗口方差"；$v_t$ 作权的双权版在吸收模型下与谱读数自洽。归属：读数。

### 2.4 白化的几何与定价：Muon、自然梯度与阻尼

**前置**：msign 恒等式 $\mathrm{msign}(M)=(MM^{\top})^{-1/2}M=U_{[:,:r]}V_{[:,:r]}^{\top}=\arg\min_{O^{\top}O=I}\Vert M-O\Vert_F^2$（Muon[^jordan24]；可扩展性版[^liu25]；矩阵函数背景 Higham 2008[^higham08]）；「流形上的最速下降」系列[^k11196][^k11215]；「Muon 赏析」[^k10592]；「为什么偏爱各向同性」[^k11549]；Polar Express（Amsel et al. 2025[^amsel25]）；谱条件缩放（Yang–Simon–Bernstein[^yang23spectral]；μP[^yang22mup]）。

**定理（Muon $=$ 谱范数球上的最速下降）**[^k11215]。解 $\max_\Phi\mathrm{tr}(G^{\top}\Phi)$ s.t. $\Vert\Phi\Vert_2\le1$：把 $G=U\Sigma V^{\top}$ 代入，$\mathrm{tr}(G^{\top}\Phi)=\sum_i\sigma_iu_i^{\top}\Phi v_i\le\sum_i\sigma_i\Vert\Phi\Vert_2\le\Vert G\Vert_\ast$，取等当且仅当 $\Phi=\sum_iu_iv_i^{\top}=\mathrm{msign}(G)$。**更新量的全部奇异值 $=1$，与 $G$ 的谱无关——谱均匀化是几何必然而非启发式**。对照：欧氏球约束给 $\Phi=G/\Vert G\Vert_F$（继承谱不均），谱范数球给全 1 谱。

**与白化/自然梯度的精确关系（三层，必须分开说）**：

1. **向量情形退化**：若把整个参数向量当作"列向量矩阵"，$\mathrm{msign}(g)=g/\Vert g\Vert$——只是归一化梯度（LARS 型），**没有白化**。白化是矩阵情形特有的。
2. **矩阵情形**：msign 在梯度矩阵自身的 SVD 基下给每个奇异方向步长 $\eta$（GD 给 $\eta\sigma_i$）。若梯度矩阵的奇异值逐层跟踪该层曲率（经验规则 $\mathbb E[gg^{\top}]\approx\sigma^2\mathcal H^2$，见「从 Hessian 近似看自适应 lr」[^k10588] 与「Muon 赏析」[^k10592]），则**逐 Fisher 模式的有效步长 $\eta\lambda_i^{\rm eff}$ 被压向均匀**——这就是"更新谱全 1"在吸收模型里的读法：$v_t\approx$常数，直到 SNR 截断。「为什么偏爱各向同性」[^k11549] 给出等价的特征层面表述：要使 SGD 成为特征层面的最速下降需 $XX^{\top}\approx I$；Muon 下参数级谱最速下降 $\approx$ 特征级谱最速下降。
3. **诚实的缺口**：上述等价需要"梯度奇异方向与 Fisher 特征方向对齐"（以及 Kronecker 型结构——K-FAC 的块结构假设，Martens & Grosse 2015[^martens-grosse-15]）假设；若不对齐，msign 只均匀化**更新**谱而非**曲率**谱。这正是 E7 要检验的内容：预条件子重整 $\{\eta\lambda_i^{\rm eff}\}$，但可吸收集合与底由 SNR 决定——**预条件子不能替代降方差**。

**白化的定价（Polar Express）**[^amsel25]。msign 用 $T$ 步多项式迭代 $X_{t+1}=a_tX_t+b_tX_t(X_t^{\top}X_t)+c_tX_t(X_t^{\top}X_t)^2$ 实现（Newton–Schulz 型，「msign 的 NS 迭代」[^k10922][^k10996]）。Polar Express 定理：**逐步贪心地解 $\arg\min_{f_t}\max_{x\in[l_t,u_t]}|f_t(x)-1|$（奇多项式，等值振荡定理定参数）即得全局最优**，收敛系数趋于 $(15/8,-10/8,3/8)$。读法：给定每步 2–3 次矩阵乘的预算，把 $[l,1]$ 谱压到 $1\pm\varepsilon_T$ 的**最小可达误差 $\varepsilon_T$ 有显式构造**——白化"恒定吸收率"的近似误差上界是有价签的；「msign（上）」[^k10922] 还给出小谱初值下限 $\sim0.1/\sqrt{\min(n,m)}$ 的灵敏度损失。

**阻尼内插（低 SNR 的代价）**。「动量的新理解」[^k11875] 把动量重释为在线岭回归 $\Phi^*=(X^{\top}X+\lambda I)^{-1}X^{\top}\partial L/\partial Y$：$Z^{-1}$ 是**阻尼 Fisher 逆**，$\lambda$ 正是 §1.3"白化 vs SNR"的折中旋钮——$\lambda\to0$ 趋自然梯度（均匀吸收、低 SNR 模式噪声爆炸），$\lambda\to\infty$ 退回普通 GD（速率 $\propto\lambda_i$）。这与 Martens 2014[^martens14] 的 damping 论证、Adam 的 $\epsilon$ 插值（「Adam ε 的缩放律」[^k10563]：$|g_i|\ll\epsilon$ 的低 SNR 分量退回线性动力学）是同一机制的三种实现。**谱上限控制**（mclip，「通过 msign 计算 mclip」[^k11006][^k11059]；Pre Decay 的谱版「坚守参数的稳定性」[^k11729]）则从顶部封住 $\eta\lambda_{\max}$ 的稳定性（edge-of-stability 上界 $2/\eta$，Cohen et al. 2021[^cohen21]）。

**尺度律**。Update RMS 对齐（Moonlight）$\mathrm{RMS}(\Phi)=\sqrt{1/\max(n,m)}$[^liu25]、官方版 $\max(1,d_{\rm out}/d_{\rm in})$ 的逐 token RMS 推导[^k11772]、谱条件 $\Vert W_k\Vert_2=\Vert\Delta W_k\Vert_2=\Theta(\sqrt{d_k/d_{k-1}})$[^yang23spectral] 共同保证"每步特征扰动恒定"——**这是均匀吸收所需步长的宽度定标**，也是优化器消融时 lr 必须按宽度律重标的原因（μP 缩放表[^k10770]）。谱读数族（Stable rank、$q{=}2$ 奇异值熵、有效秩；Roy & Vetterli 2007[^roy07]；「矩阵的有效秩」[^k10847]、「奇异值熵」[^k11767]）提供低成本的谱压缩监控量。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **旋钮 1 的连续化**：冻结态窗口内把更新改为阻尼白化 $\delta=-\eta(F+\lambda_{\rm damp}I)^{-1}g$（实现用「动量的新理解」的 $Z^{-1}M$ 形式或 SOAP 型结构化近似[^vyas24]），扫 $\lambda_{\rm damp}$——**不在 {SGD, Adam, Muon} 里离散选优化器，而是在单参数族上连续调**。
2. **输出层单独处理**：lm_head 按逐列 normalized SGD（步长 $\eta/d$、初始 $\sigma=\Theta(1/d)$，[^k11647]），避免大词表层的谱各向异性污染全局读数。
3. **在线谱测量**：流式幂迭代（每步一次 QR，误差不累积[^k11710]）在线测 $b=\mathbb E[J^{\top}Fr]$ 的主子空间与 $\Sigma_t$；msign 用 Polar Express 系数（按预算定价 $\varepsilon_T$）；mclip 封顶防 $\eta\lambda_{\max}$ 过冲。
4. **谱读数族进日志**：Stable rank、$q{=}2$ 奇异值熵、各向异性度（[^k10847][^k11767][^k11772]）。

*理论收益：*

- **命题 2.4.1（阻尼白化的速率族）**：对角 Fisher 模型 $+$ Tikhonov 阻尼下，逐模式
  $$\eta\lambda_i^{\rm eff}(\lambda_{\rm damp})=\eta\,\frac{\lambda_i}{\lambda_i+\lambda_{\rm damp}}$$
  （推导：$(F+\lambda_{\rm damp}I)^{-1}g$ 在特征基下的分量）。$\lambda_{\rm damp}\to0$ 得恒定吸收（自然梯度，$v_t=\eta\cdot\text{const}$）；$\to\infty$ 得 GD 谱（$\propto\lambda_i$）；中间值给出 $d(t)$ 的**单参数形状族**。Adam 的 $\epsilon$（[^k10563]）是它的逐坐标全局版。
- **命题 2.4.2（白化的底定价）**：同一参数下逐模式 OU 平衡给
  $$d_\infty(\lambda_{\rm damp})=\frac{\eta}{4B}\sum_i\frac{\sigma_i^2}{\lambda_i+\lambda_{\rm damp}}+d_{\rm gap}$$
  （推导：收缩率 $a_i=\eta\lambda_i/(\lambda_i+\lambda_{\rm damp})$、注入方差 $\eta^2\sigma_i^2/(B(\lambda_i+\lambda_{\rm damp})^2)$，代入 $s^2/2a$，曲率不再完全消掉）。$\lambda_{\rm damp}\to0$ 时低 $\lambda$ 模式底爆炸——"**无法白化测不准的方向**"（[^martens14]）的闭式形式；$\to\infty$ 退回 SGD 底。**收益**：吸收均匀度（旋钮 1）与底膨胀（旋钮 3）被一个参数显式权衡，形状族与底族都可闭式拟合——优化器比较从离散选择升级为沿参数族测形状。
- **命题 2.4.3（输出层定标）**：LM head 的匹配更新是逐列 normalized SGD，跨词表/宽度按 $\Theta(1/d)$ 缩放（[^k11647]）——OPD 大词表层的 $\eta$ 缩放律。

### 2.5 幂律吸收曲线的起源：$\alpha=\gamma/\beta$ 与谱尾反演

**前置**：量子化缩放假设（Michaud et al. 2023[^michaud23]；「量子化假设推 Scaling Law」[^k9607]）；NTK 谱按频率多项式衰减与谱偏差/F-Principle（Basri et al. 2019[^basri20]；Rahaman et al. 2019[^rahaman19]；Xu et al. 2020[^xu20]）；核回归梯度流的学习曲线（Bordelon, Canatar & Pehlevan 2020[^bordelon20]；Canatar et al. 2021[^canatar21]；早停即谱正则化：Yao et al. 2007[^yao07]、Dieuleveut & Bach 2016[^dieuleveut16]）；T3（§1.4）。

**推导**。设监督信号由"量子"组成，出现频率服从 Zipf：$p_k=k^{-(\gamma+1)}/Z_\gamma$（$\gamma>0$；quanta 假设的实证基础见[^michaud23]）。谱吸收侧把量子映射到模式：第 $k$ 个量子的振幅 $a_k\propto p_k$（频率 $=$ 监督质量），曲率按谱偏差标度 $\lambda_k\propto k^{-\beta}$（$\beta>0$：越细粒度/越高频的模式曲率越低）。代入 $d(t)-d_\infty=\sum_ka_ke^{-\eta\lambda_kt}$，把对 $k$ 的求和换成对 $u=\lambda_k=k^{-\beta}$ 的积分：$k=u^{-1/\beta}$、$|dk|=\frac1\beta u^{-1/\beta-1}du$，于是

$$
d(t)-d_\infty\ \propto\ \int u^{(\gamma+1)/\beta}e^{-\eta tu}\cdot\frac1\beta u^{-1/\beta-1}\,du
=\frac1\beta\int u^{\gamma/\beta-1}e^{-\eta tu}\,du
=\frac{\Gamma(\gamma/\beta)}{\beta}\,(\eta t)^{-\gamma/\beta}.
$$

**结论（E1）**：只要"振幅幂律 $\times$ 曲率幂律"，吸收曲线必是幂律，且

$$\alpha=\frac\gamma\beta,\qquad \rho(\lambda)\propto\lambda^{\alpha-1}\ \ (\lambda\to0),$$

后半是 Laplace 变换的 Abelian 定理（T3，Feller[^feller71]）。**实测 $\alpha\in[0.44,0.67]$ 直接反演谱尾指数，无需数值反演**——NNLS 谱反演因此有闭式对照。三个推论：

1. **lr 折叠自动成立**（只通过 $\eta t$ 进入）；**数据规模无关自动成立**（数据改振幅总量与截断点，不改 $\gamma,\beta$）。
2. **平台**：quanta 侧阈值 $p_k<\tau/D$（一个量子至少 $\tau$ 个样本才可学，[^k9607]）与优化侧 $\lambda_{\rm crit}$ 给幂律两个下截断 $\Rightarrow d_\infty$；二者对应同一段谱的不同截断机制（数据供给 vs 信噪比），可分离。
3. **可证伪点**：若谱是对数正态/分层混合而非幂律，大 $t$ 行为应为 stretched exponential 而非幂律（T4，[^palmer84]）。核回归一系（[^bordelon20][^canatar21][^yao07]）已对"幂律谱 $\Rightarrow$ 幂律学习曲线"给出严格版本，本条是其在"监督谱 $=$ 量子频率"假设下的 OPD 特化。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **反演顺序倒转**：先按 T3 闭式从实测 $\alpha$ 读谱尾（$\rho\propto\lambda^{\alpha-1}$），NNLS 反演降级为残差检查与形状分类（幂律/带边/stretched）。
2. **数据侧按量子频率分配**：监督（token/题目）按 $p_k$ 排序与加权，使可吸收集最大化；数据规模实验的解释框架固定为"只移截断点、不改 $\gamma,\beta$"。
3. **事前审计**：用教师-学生 advantage 分布估 $\sigma_i/c_i$，把 $\lambda_{\rm crit}$ 的两个截断（数据侧 $\tau/D$、优化侧 SNR）合并成一个预算公式，事前标记"喂多少数据都吃不到"的监督段。

*理论收益：*

- **命题 2.5.1（$\alpha$ 的三重身份）**：$\alpha=\gamma/\beta$ 同时是 (i) 吸收曲线指数、(ii) 谱尾指数、(iii) quanta 频率尾/曲率标度比——三个可独立测量的量被一个等式锁定，任意两个对上第三个即模型的强证据。
- **命题 2.5.2（OPD 的底缩放律）**：被吸收模式集为 $k\le\min\{k_{\rm data},k_{\rm opt}\}$，其中
  $$k_{\rm data}\sim\Big(\frac{D}{\tau Z_\gamma}\Big)^{\frac1{\gamma+1}}\ \ (\text{数据侧：}p_k\ge\tau/D),\qquad
  k_{\rm opt}\sim\Big(\frac{\lambda_1}{\lambda_{\rm crit}}\Big)^{\frac1\beta}\ \ (\text{优化侧：}\lambda_k\ge\lambda_{\rm crit}),$$
  由 Zipf 尾部积分 $\sum_{k>K}p_k\approx K^{-\gamma}/(\gamma Z_\gamma)$ 得
  $$d_\infty\ \propto\ \big[\min(k_{\rm data},k_{\rm opt})\big]^{-\gamma}+d_{\rm gap}.$$
  **收益**：数据规模与 $(\eta,B)$ 的边际收益曲线闭式可预测——$D\to\infty$ 时底被优化侧钉住（"data-overfed, algorithm-starved"的量化形式）；反之 $B\uparrow/\eta\downarrow$ 时底由数据侧钉住。哪侧在钉，直接决定该往哪个旋钮花钱。
- **命题 2.5.3（形状分类学）**：幂律 / 带边弯曲 / stretched exponential $\leftrightarrow$ 纯幂律谱 / 双截断谱 / 对数正态分层谱（T4，[^palmer84]）——形状判别即谱类型判别，给谱反演一个先验分类。

### 2.6 Adam 的两个软底（A4 的定量化）

**前置**：Adam（Kingma & Ba 2015[^kingma15]）；sign 解剖（Balles & Hennig 2018[^balles18]）；signSGD（Bernstein et al. 2018[^bernstein18]）；Adam 末期病态（Reddi et al. 2018[^reddi18]）；rotational equilibrium（Kosson et al. 2023[^kosson23]）；Surge（[^li24surge]）；「Update RMS 0.2」[^k11267]、「Weight RMS 渐近」[^k11307][^k11404]、「从 Hessian 近似看 Adam」[^k10588]、「$\beta_1=\beta_2$ 最优」[^k11593]。

**推导一（Update RMS 的平均场）**。$u=m/\sqrt v$；用 $\mathbb E[u^2]\approx\mathbb E[m^2]/\mathbb E[v]$（忽略 $\mathrm{Cov}(m^2,v)$ 的平均场近似[^k11267]）。$\mathbb E[m^2]=\mu^2+\mathrm{Var}(m)=\mu^2+\kappa\sigma^2$（§2.1 的 $\kappa=\frac{1-\beta_1}{1+\beta_1}$）；$\mathbb E[v]=\mu^2+\sigma^2$（$g$ 的二阶矩无偏）。故

$$
\mathbb E[u^2]\approx\frac{\mathrm{SNR}+\kappa}{\mathrm{SNR}+1},\qquad \mathrm{SNR}=\frac{\Vert\mu\Vert^2}{\Vert\sigma\Vert^2},\qquad
\kappa=\frac{1-\beta_1}{1+\beta_1}.
$$

**同一个 $\kappa$ 的第三次出现**：它是 $\beta_1$-EMA 对白噪声的方差压缩因子——(i) 动量噪声压缩（§2.1），(ii) Adam 低 SNR 更新 RMS$^2=\kappa$（$\beta_1=0.9$：RMS $=0.229$），(iii) Weight RMS 公式（[^k11307]）里的噪声底。一个常数贯穿三处，因为它只依赖 EMA 的记忆结构。

**推导二（Adam 的软底）**。低 SNR（$\lambda<\lambda_{\rm crit}$ 的坐标）时 $\mathbb E[u^2]\to\kappa$：**即使真实梯度只剩噪声，Adam 仍以每坐标 RMS $\approx0.229\eta$ 随机游走**。把它作为逐坐标 OU 的注入噪声（$x_{t+1}=(1-\eta\lambda)x_t+\eta u_t$，$u_t\approx\mathcal N(0,\kappa)$）：

$$
\mathbb E[x^2]_\infty^{\rm Adam}=\frac{\eta^2\kappa}{1-(1-\eta\lambda)^2}\approx\frac{\eta\kappa}{2\lambda}
\quad\Longrightarrow\quad
d_\infty^{\rm Adam}\approx\frac{\eta\,\kappa}{4}\,N_{\rm low\text{-}SNR},
$$

其中 $N_{\rm low\text{-}SNR}$ 是信噪比不足的坐标数。**三个关键性质**：(i) 逐坐标曲率 $\lambda$ 在分子分母消掉——底是**平的**（比 SGD 底更平：SGD 底逐模式权 $\sigma_i^2$，Adam 软底逐坐标等权）；(ii) **与 $B$ 无关**（sign 噪声不随 batch 平均——与 SGD 底 $\frac{\eta}{4B}\sum\sigma_i^2$ 的本质区别）；(iii) 随 $\beta_1\to1$ 减小（$\kappa\to0$：$\beta_1=0.98$ 时 $\kappa=0.01$，底缩 5 倍）。与 weight decay 并存时，[^k11307] 的定态 $\Vert\theta\Vert_{\rm RMS}\approx\sqrt{\eta/2\lambda_{\rm wd}}$ 给第三个定态量（$\lambda_{\rm wd}=0$ 时按 $\eta\sqrt t$ 膨胀）。**双底分解（E2）**：

$$d_\infty\ =\ d_{\rm gap}+\underbrace{\tfrac{\eta}{4B}\sum_i\sigma_i^2}_{\text{SGD/梯度噪声底（}\propto1/B\text{，权 }\sigma_i^2\text{）}}+\underbrace{\tfrac{\eta}{4}\cdot\tfrac{1-\beta_1}{1+\beta_1}\,N_{\rm low\text{-}SNR}}_{\text{Adam 软底（}B\text{ 无关，平权）}}.$$

**相关事实的归位**：$\beta_1=\beta_2$ 使 $\hat v_t=\hat m_t^2+\hat\sigma_t^2$（方差分解精确、$\hat v\ge\hat m^2$；极小极大证明见[^k11593]）——把 $v$ 从二阶矩变成方差估计，即 A3"真 Fisher 口径"在优化器内部的对应物；Surge（最优 lr 随 $B$ 非单调）来自 $\mathbb E[u^2]$ 对 SNR 的依赖与 $B$ 增大的竞争[^li24surge]；$\beta_2$ 的 EMA 记忆 $\approx1/(1-\beta_2)=1000$ 步与 run 同量级 $\Rightarrow$ 分母跟不上信号衰减，在谱尾部垫一层软底（[^reddi18] 末期病态的近亲；源头文档 H3 的候选机制之一）。隐式梯度正则（有限 lr $=$ 修正 ODE 的梯度惩罚 $\frac\eta4\Vert\nabla L\Vert^2$，Barrett & Dherin 2020[^barrett20]；「为什么 lr 不宜过小」[^k7787]）提醒：**降 $\eta$ 压底的同时丢失趋平坦偏置**——lr 的三重角色见 E10。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **$(\beta_1,B)$ 联合底预算**：底预算 $d_\infty^{\rm target}$ 给定后，在 $(\beta_1,B)$ 平面解最便宜的组合——$\beta_1$ 免费（只改记忆）、$B$ 昂贵（改显存/吞吐），故先压 $\kappa(\beta_1)$ 再动 $B$。
2. **在线 SNR 读数**：按[^k11267] 从 update RMS 反解 $\Vert\mu\Vert/\Vert\sigma\Vert$（$\mathrm{SNR}=\frac{\kappa-\Vert u\Vert^2}{\Vert u\Vert^2-1}$ 的反函数），逐层监控 $N_{\rm low\text{-}SNR}(t)$——软底分量的实时估计，零额外开销。
3. **正交消融预登记**：$\beta_1$ 扫描（理论上只动软底）$\times$ $B$ 扫描（理论上只动 SGD 底）——一次实验分离 E2 的三个成分。

*理论收益：*

- **命题 2.6.1（软底公式）**：$d_\infty^{\rm Adam}=\frac{\eta\kappa(\beta_1)}4\,N_{\rm low\text{-}SNR}$——$B$ 无关、谱平权、$\kappa$ 律（$\kappa$ 从 $1$（$\beta_1{=}0$）缩到 $0.01$（$\beta_1{=}0.98$），软底缩 99 倍）。
- **命题 2.6.2（双底的线性可分性）**：三成分在 $(\beta_1,B)$ 扫描平面上的响应方向两两正交：软底只随 $\beta_1$（$\kappa$ 律）、SGD 底只随 $B$（$1/B$ 律）、两者都不动的残差即 $d_{\rm gap}$（口径/容量决定）——**底成分分解从判据变成实验代数**，且与命题 2.1.1/2.1.3 的 EMA 分解交叉验证。

### 2.7 距离口径与底的标度（D2 的解析裁决）

**前置**：正态族三口径闭式（「KL/巴氏/W 距离」[^k8512]）；「GlobalPointer 下的 KL」[^k9039]（KL 的 logit 空间表示 $D(p,q)=\langle f(s)-f(t),\,s-t\rangle$，即 Fisher 度量的显式形）；L1 口径来自论文读数 $\overline{|A|}$[^fu26]。

**推导**。底层物理量只有一个：每模式定态方差 $\mathbb E[x_i^2]_\infty\propto\eta/B$（§1.3）。不同读数口径是对同一 $\{x_i\}$ 的不同泛函，幂次由泛函的阶决定：

- **KL/二次型口径**：$d_\infty^{\rm KL}=\frac12\sum_i\lambda_i\mathbb E[x_i^2]_\infty\propto\eta/B$（曲率消掉，权 $\sigma_i^2$）；
- **L1/线性口径**：$\mathbb E|x_i|_\infty=\sqrt{2/\pi}\,\big(\mathbb E[x_i^2]_\infty\big)^{1/2}\propto\sqrt{\eta/B}$，逐模式权 $\propto\sqrt{\lambda_i}\,\sigma_i/\sqrt B$——**平坦模式主导**；
- 一般 $p$ 阶泛函 $d_\infty^{(p)}\propto(\eta/B)^{\min(p,2)/2}$。

**静态版对照**（正态族，[^k8512]）：$\mathrm{KL}(p\|q)=\frac12\big[\Delta\mu^{\top}\Sigma_q^{-1}\Delta\mu+\mathrm{Tr}(\Sigma_q^{-1}\Sigma_p)-\log\det(\Sigma_q^{-1}\Sigma_p)-n\big]$（$\Sigma_q^{-1}$ 加权的二次型 $+$ log-det——偏袒高曲率方向）；$\mathcal W_2^2=\Vert\Delta\mu\Vert^2+\Vert\Sigma_p^{1/2}-\Sigma_q^{1/2}\Vert_F^2$（可交换时；协方差差的"开平方后作差"——线性泛函）；巴氏居中。**同一对分布、三种曲率结构**。这正是源头文档推导八"KL 代理 vs 论文 L1 读数"的测度论根源，也解释了为什么两个口径的谱反演各自自洽（口径依赖）。

**口径非本质判据（E8）**：WGAN 的有效成分是 critic 的 Lipschitz 约束而非 W 泛函本身（Fedus et al. 2018[^fedus18]；WGAN-GP 从稳定性公理导出而未用 W 距离，Gulrajani et al. 2017[^gulrajani17] 与「互怼的艺术」[^k4439]；KR 对偶的曲率全部来自约束集几何，Villani 2008[^villani08] 与「W 对偶到 WGAN」[^k6280]；f-散度族效果差别不大，Nowozin et al. 2016[^nowozin16] 与「f-GAN」[^k6016]）。合并成一句：**只要两个口径诱导的曲率结构（Hessian 谱）相同，吸收节奏应相同；反之，KL 与 L1 的谱反演差异应可由本节的幂次公式吸收**。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **多口径并行读数**：逐 token 日志同时输出 L1/KL/CE 三个口径 $+$ 一个参数 RMS 口径；跨口径比较只经命题 2.7.1 换算，不直接比数值。
2. **距离口径升级（可选）**：若读数方差过大，换巴氏/$\mathcal W_2$ 型对称口径（[^k8512] 的正态族对照显示其 $\lambda$-加权介于 KL 与 L1 之间）；改口径 $=$ 改 $d(t)$ 的模式分解（A5），谱反演按口径换算。
3. **目标侧保持不动**：reverse-KL 的 score 依赖 $p$ 本身、不满足可采样公理（[^k11854]），OPD 维持 CE/advantage 型代理 $+$ 中心化；§1.1(b) 的曲率修正项 $C$ 作为**已知偏差记录**，不试图在目标里修掉。

*理论收益：*

- **命题 2.7.1（口径换算律）**：$d_\infty^{(p)}\propto(\eta/B)^{\min(p,2)/2}$，且逐模式权随阶数从 $\sigma_i^2$（二次泛函）变到 $\sqrt{\lambda_i}\,\sigma_i$（线性泛函）——**幅度标度与谱加权的双重换算公式**（L1 $\leftrightarrow$ KL），D2 的两分支由此统一。
- **命题 2.7.2（口径不变性判据）**：同曲率结构的口径预言同一 $v_t$ 形状；按 2.7.1 换算后的残差就是读数伪影（KL 的熵污染、CE$-$H 相减伪影）的定量证据——A5 的"跨口径只读方向不读数值"升级为可计算。

### 2.8 蒸馏目标的恒等式改写（降方差不改不动点）

**前置**：得分匹配 $=$ 条件得分匹配（Vincent 2011[^vincent11]）；SiD（Zhou et al. 2024[^zhou24sid]）与「基于恒等式的蒸馏」[^k10085][^k10567]；MeanFlow（Geng et al. 2025[^geng25meanflow] 与「从瞬时速度到平均速度」[^k10958]）；一致性模型（Song et al. 2023[^song23cm]；sCM，Lu & Song 2024[^lu24scm]）；shortcut（Frans et al. 2024[^frans24]）；扩散 ODE/SDE 框架（[^song21sde][^song21ddim]；v-参数化 Salimans & Ho 2022[^salimans22]；EDM 的 SNR 预条件 Karras et al. 2022[^karras22]）。

**三个模板**：

1. **换表示不改不动点（Vincent）**。条件得分匹配与（边际）得分匹配的最优解相同，因为两损失之差

$$\mathbb E_{x_t}\Big[\mathbb E_{x_0|x_t}\Vert\nabla\log p(x_t|x_0)\Vert^2-\Vert\nabla\log p(x_t)\Vert^2\Big]$$

不含 $\theta$（展开平方、交叉项用全期望公式消去）。**判据**：任何改写只要与原目标差一个 $\theta$-无关项，不动点不动——这是 OPD 改写 advantage 表示（immediate vs 后缀和）的合法性检验。

2. **条件期望可被单样本替换（SiD）**。若 $\epsilon_{\varphi^*}=\mathbb E[\varepsilon|x_t]$，则对任意只依赖 $(x_t,t)$ 的 $f$：$\mathbb E[\langle f,\epsilon_{\varphi^*}\rangle]=\mathbb E[\langle f,\varepsilon\rangle]$（全期望公式一步）。于是蒸馏损失 $\mathbb E\Vert\epsilon_{\varphi^*}(x_t^{(g)})-\epsilon_{\psi^*}(x_t^{(g)})\Vert^2$ 恒等变形为 $\mathbb E[\langle\epsilon_{\varphi^*}-\epsilon_{\psi^*},\ \epsilon_{\varphi^*}-\varepsilon\rangle]$——把"需要两个最优网络"换成"需要最优教师 $+$ 单样本噪声"；交叉项显式化后可用 $\mathcal L_4^{\rm(sg)}=2\mathcal L_2^{\rm(sg)}-\mathcal L_1^{\rm(sg)}$ 在**梯度层面**恒等（stop-gradient 模板，[^k10567]）。**警告**（MeanFlow 的 sg 分析，[^k10958]）：sg 使驻点未必是极小点——等效损失与优化器动力学耦合，其 Hessian 不再是 Fisher。

3. **区间平均替代瞬时量（MeanFlow）**。$u(x_t,r,t)=\frac1{t-r}\int_r^tv\,\mathrm d\tau$ 满足恒等式 $v=u+(t-r)\frac{d}{dt}u$；把逐时刻目标换成区间平均目标后，大步长不再是近似误差而是恒等约束。对 OPD 的对应：把逐时刻 $v_t$ 换成区间平均 $\bar v_{[t_1,t_2]}=\frac{d(t_1)-d(t_2)}{\int_{t_1}^{t_2}\mathrm dt}$（§2.3 恒等式的连续版），读数噪声随之下降。

**移植到 OPD 的结构对应与缺口**。对应：OPD 的 advantage $A_i$ 是中心化策略梯度家族的打分函数（§1.1(b)），基线/中心化自由度即方差工程（最优 baseline $\propto\mathbb E[A\Vert\nabla\log\pi\Vert^2]$，[^k7737] 与[^greensmith04]）；完整梯度含后缀和 $\sum_{k\ge t}A_k$，它是"未来 token 条件分布上的期望"的蒙特卡洛估计。缺口：SiD 的替换之所以可行，是因为被替换项（教师得分）**本身是某条件期望**；OPD 的教师项给定 $(s,y)$ 是精确值、方差在学生侧的后缀和上——直接移植需要先把后缀和表示为某个条件期望（例如经序列 KL 的链式分解），这一步目前开放。可行的第一步是控制变量式校正（命题 2.8.1 给出其收益上界与最优形式），"退火到无偏"成为约束而非经验（EMO 的反例说明不加约束会移动不动点，[^k9797]）。

**OPD 视角：改什么、拿到什么**

*改动：*

1. **两步目标改写**：第一步（安全）：控制变量式 advantage 校正 $A\to A-\beta^{\top}(Z-\mathbb EZ)$，$Z$ 取可控特征（逐 token 曲率、教师熵、位置、温度），$\beta^*=\mathrm{Cov}(A,Z)\,\mathrm{Var}(Z)^{-1}$；第二步（需验证）：序列 KL 链式分解后检验后缀和 $\sum_{k\ge t}A_k-\text{centering}$ 是否为"给定前缀"的条件期望，若是，用 FGM 模板 $\mathcal L_4^{\rm(sg)}=2\mathcal L_2^{\rm(sg)}-\mathcal L_1^{\rm(sg)}$ 做单样本替换 $+$ sg 度规位（[^k10085][^k10567]）。
2. **上线护栏**：sg 改写后等效 Hessian 不再是 Fisher（[^k10958] 的驻点警告）——上线条件：toy 序列上数值验证梯度等价 $+$ $\lambda_{\max}(\nabla^2J')\le2/\eta$。
3. **基线工程**：中心化升级为曲率加权最优 baseline（[^k7737][^greensmith04]）。

*理论收益：*

- **命题 2.8.1（方差缩减的控制变量上界）**：可控特征集 $Z$ 下，任何保持梯度的改写其逐 token 方差满足 $\mathrm{Var}(\hat A)\ge\mathrm{Var}(A)(1-R^2)$，$R^2$ 为 $A$ 对 $Z$ 回归的判定系数；最优改写即控制变量形式 $\hat A^*=A-\beta^{*\top}(Z-\mathbb EZ)$（标准控制变量理论；E9 的乘性加权——focal/TaiLr/EMO——只是其特例）。**收益**：P4 的理论收益上限由 $R^2$ 给出且可直接从日志估计——先测 $R^2$ 再决定是否值得做加权。
- **命题 2.8.2（后缀和 $=$ 未用控制变量）**：immediate-token 近似与完整后缀和之差恰是"未来条件分布"这组控制变量；把后缀和表示为条件期望后，SiD 恒等式允许单样本替换，不动点不动（Vincent 判据，[^vincent11][^k9509]），方差收益受 2.8.1 的上界约束。
- **命题 2.8.3（改写的可上线条件）**：梯度等价（$\nabla J'=\nabla J$，不动点不变）$+$ $\lambda_{\max}(\nabla^2J')\le2/\eta$（等效损失在当前优化器下稳定）——把 MeanFlow 的 sg 驻点警告（[^k10958]）算子化为两条可检验判据。

---

## 3. 理论拓展命题（E1–E10）

> 标注：★$=$ 本轮新推导；☆$=$ 文献结果 $+$ OPD 化。§2 各节末的**命题 2.x.n** 是局部结果（OPD 记号下的具体形式）；本节 **E1–E10** 是跨节综合命题——两套编号互引，不重复陈述。

**E1（★，谱尾指数 $=$ 吸收指数）**。§2.5：$\alpha=\gamma/\beta$，$\rho(\lambda)\propto\lambda^{\alpha-1}$。把 brief §2.2"定量形状携带谱结构信息"升级为闭式反演。方法学先例：核回归学习曲线[^bordelon20][^canatar21] 与 Tauberian 定理[^feller71]。

**E2（★，双底分解）**。§2.6：$d_\infty=d_{\rm gap}+\frac{\eta}{4B}\sum\sigma_i^2+\frac{\eta\kappa}{4}N_{\rm low\text{-}SNR}$。两个底的谱加权（$\sigma_i^2$ vs 平权）与 $B$ 标度（$1/B$ vs 无）不同，原则上可分离；若两底都远小于观测平台，则 $d_{\rm gap}$（表示缺口）主导，方差工程收益封顶。

**E3（★，口径–标度对应）**。§2.7：$d_\infty^{(p)}\propto(\eta/B)^{\min(p,2)/2}$。D2 的两分支是读数口径而非物理差异。

**E4（☆，EMA/Polyak $=$ 免费的终点退火）**。§2.1：「炼丹七」[^k11804] 定理 $+$ SWA/Polyak[^izmailov18][^polyak92] $+$ Schedule-Free[^defazio24]。三旋钮中唯一"只动 $d_\infty$、完全不动 $\eta\lambda^{\rm eff}$ 谱"的干预；「动力学（四）」的线性化分析（$\ddot\theta\approx-\theta/4$，零阻尼振荡[^k6583]）提示它同时消除动量欠阻尼振荡。

**E5（☆→★，动量阻尼 $=$ 一阶模型的第一阶畸变）**。「动力学（一）」[^k5655]：动量 $=$ 蛙跳离散的 $\ddot\theta+\lambda\dot\theta=-\nabla L$[^sutskever13]，逐模式欠阻尼时 $x_i\propto e^{-\lambda t/2}\cos\omega_i t$、$\omega_i=\sqrt{\lambda_i-\lambda^2/4}$。推论：衰减率减半且 $v_t$ 被振荡调制——KL 口径 step$\sim$55–80 的凸起（源头文档 H3a）若为 $\beta_2$ 记忆欠阻尼瞬态，应随 $\beta_2$ 降低周期缩短、随 EMA 消失；与 E2 的软底一起构成 Adam 畸变的完整清单。

**E6（☆，可吸收模式数的容量三角）**。「词向量维度」[^k7695] 的 $n>8.33\log N$（Laplace 渐近）与 JL 引理 $n\gtrsim c\,\varepsilon^{-2}\log N$（Dasgupta & Gupta 2003[^dasgupta03]）合流：可吸收模式数 $N_{\rm abs}$、表示维度 $n$、精度 $\varepsilon$ 满足 $n\gtrsim c\,\varepsilon^{-2}\log N_{\rm abs}$。"$\lambda<\lambda_{\rm crit}$ 吃不到"与"维度/样本不足以分辨的模式不存在"是同一枚硬币（有限分辨率下的不可分辨性）；谱集中度读数用 $S^*_{1,2}=\frac{\mathbb E|x|}{\sqrt{\mathbb E[x^2]}}$（稀疏度公理体系 Hurley & Rickard 2009[^hurley09]；「数据稀疏度量」[^k9595]）。

**E7（☆，更新谱与曲率谱解耦）**。§2.4：「Muon $=$ 谱范数最速下降」[^k11215] $+$ 最优预条件 $H^*=\Sigma^{1/2}$（「炼丹九」[^k11882]；PSGD 系 Xi-Lin Li 2018[^li18psgd]）$+$ msign 后 lr–$B$ 律不变[^k11285]。定理化陈述：**预条件子只能重整更新谱 $\{\eta\lambda_i^{\rm eff}\}$；一个模式能否被吸收（$\lambda_{\rm crit}$）与终态底由 SNR 与方差决定**。这是 brief §4 反直觉推论 2 的定理形式，也是 M6 的判据（$\alpha$ 跨优化器不变 $\Rightarrow$ 几何性质，H6）。

**E8（☆，口径非本质判据）**。§2.7 末：[^fedus18][^gulrajani17][^nowozin16][^k8244] 合并——口径族内（同曲率结构）吸收节奏不变；跨曲率结构（KL 的 log 项 vs L1 的符号）按 E3 换算。

**E9（☆→★，恒等式改写空间与最优加权）**。§2.8：梯度等价类 $\{J':\nabla J'=\nabla J\}$ 内选方差最小者；统一语法 $\nabla_zL=W(p)\cdot(q-e_i)$（CE 的 $W=I$；Brier/Tsallis 的 $W=2(\mathrm{diag}(q)-qq^{\top})$ 即 Fisher 本身；focal 的 $W=(1-p)^\gamma$；TaiLr 的有理权——「LM loss 的选择」[^k11854]、「过度自信」[^k9526]、「focal」[^k4733]、「EMO」[^k9797]）。约束 $\mathbb E[WA]=\mathbb E[A]$ 使"退火到无偏"内生化。

**E10（☆，lr 的三重角色与最优内点）**。合并四个约束：(i) 吸收速率 $\eta\langle\lambda\rangle$（小 $\eta$ 慢）；(ii) 噪声底 $d_\infty\propto\eta$（大 $\eta$ 抬底）；(iii) $\lambda_{\rm crit}\propto\sqrt\eta$（**大 $\eta$ 反而吃不到低曲率模式**）；(iv) 稳定性 $\eta\lambda_{\max}^{\rm eff}\le2$（edge-of-stability[^cohen21]）与隐式梯度正则 $\frac\eta4\Vert\nabla L\Vert^2$（小 $\eta$ 丢偏置，[^barrett20][^k7787]）。四个约束把 $\eta$ 夹进一个区间，最优 $\eta$ 是内点。这把"lr 只重标时间轴"（经验事实 3：谱形状族）与"最优 lr 存在"（端点选择）的表面张力显式化：折叠律描述族内形状，内点描述族内选点。

---

## 4. 与既有实验程序的衔接

三句话（细节仍在源头文档 §6）：(i) §2.1–2.3 的三个读数工具（$B_{\rm eff}$ 折算、终点恒等式、尾窗降噪）可直接并入 M3 日志管线与 D2 裁决；(ii) §2.4 的流式幂迭代（「流式幂迭代的 Muon 实现」[^k11654]–[^k11710]：每步一次 QR、SCQR 误差不累积定理、免费输出 $\Sigma_t$ 与谱范数读数）是 M5 冻结态谱测量的在线替代/补充——样本量按随机 SVD 与 JL 定（Halko et al. 2011[^halko11]；[^dasgupta03]），随机矩阵谱范数零假设 $\mathbb E\Vert W\Vert_2\approx\sqrt n+\sqrt m$（Davidson–Szarek[^davidson01]；「快速估计谱范数」[^k11335]）作对照基线；(iii) §2.2/2.6 的公式在 M4/M6 消融前预登记为可证伪预测（$\beta_1$ 扫描的 $\kappa$ 律、宽度律重标的 lr）。

---

## 附录 A：理论资产总库

> 体例：**ID**（主题，年份）＋脚注 — 核心结果（公式忠实原文）。

### A.1 收敛理论与学习率调度（「让炼丹更科学」系列）

- **9902**（2023）[^k9902] — 投影 SGD 的 regret 界 $\frac1T\sum_tE[L(\theta_t)-L(\theta^*)]\le\frac{R^2}{2T\eta_T}+\frac{G^2}{2T}\sum\eta_t$：**右端 $=$ 瞬态 $+\propto\eta$ 的底**（T1 原型）。
- **11469**（2025）[^k11469] — 无界域版 $\sum_t\eta_tE[L(\theta_t)-L(\varphi)]\le\frac{\Vert\theta_1-\varphi\Vert^2}2+\frac{G^2}2\sum\eta_t^2$；收敛条件 $\sum\eta_t=\infty,\ \sum\eta_t^2/\sum\eta_t\to0$。
- **11480**（2025）[^k11480] — 终点恒等式（§2.3）；终点损失 $\mathcal O(\sqrt{\ln T/T})$；质心 $\bar\theta_T$ 以平均损失速率收敛（Jensen）。
- **11494**（2025）[^k11494] — 加权恒等式 $+$ 线性衰减最优性（§2.2）；$\eta_t=\alpha(1-\frac t{T+1})$ 达信息论下界。
- **11530/11540**（2026）[^k11530][^k11540] — 闭环最优 lr $\eta_t=\frac{RG_t^{-2}}{\sqrt{Q_T}}(1-\frac{Q_t}{Q_T})$ 及其严格构造证明（主项常数 $1/4$）；因果版解释 Warmup。
- **11804**（2026）[^k11804] — 双权恒等式：权重平均 $\equiv$ 终点效果 $\times\frac{v_{t+1:T}}{v_{1:T}}$；EMA $\approx$ 提前 $\mathcal O((1-\gamma)^{-1})$ 步窥见线性衰减；Schedule-Free 分析[^defazio24]。
- **11879**（2026）[^k11879] — 多阶段 lr：等效步数 $T_2^\#=\frac12T_2+\frac12\sqrt{T_1(2T_2-T_1)}$；计划式 minimax 的球交几何闭式。
- **11882**（2026）[^k11882] — Mahalanobis 恒等式 $\Rightarrow$ 最优预条件 $H^*=\Sigma^{1/2}$（梯度白化）；AdaGrad[^duchi11]/RMSProp/Adam[^kingma15]/Shampoo[^gupta18]/SOAP[^vyas24] 皆为其结构化近似；EMA 二阶矩破坏单调性 $=$ 理论实践分叉点。
- **7787**（2020）[^k7787] — 有限 lr $=$ 修正 ODE 的精确解：$\tilde g=\nabla\big(L+\frac\gamma4\Vert\nabla L\Vert^2\big)$（隐式梯度正则[^barrett20]）。

### A.2 lr × batch × 缩放律 × Adam 渐近

- **11260**（2025）[^k11260] — $\eta^*\approx\frac{\eta_{\max}}{1+\mathcal B_{\rm noise}/B}$（$\mathcal B_{\rm noise}$ 记号承 McCandlish et al. 2018[^mccandlish18]）；数据-步数律 $(\frac S{S_{\min}}-1)(\frac E{E_{\min}}-1)=1$；SDE 视角[^li17sme]、batch–lr 实证[^smith18decay]。
- **11280**（2025）[^k11280] — $\mathbb E[\mathrm{sign}(\tilde g_B)]\approx\frac g{\sqrt{g^2+\sigma^2/B}}$；进度公式对预条件齐次（普适），$\eta^*$ 不齐次；Surge 现象[^li24surge]。
- **11285**（2025）[^k11285] — $\mathbb E[\tilde\Phi_B]\approx\mathrm{msign}(G)(1+\mathcal B_{\rm simple}/B)^{-1/2}$：msign 输出正交 $\Rightarrow$ lr–$B$ 律与 SignSGD 同形。
- **11301**（2025）[^k11301] — 动量等效 batch $\times\frac{1+\beta_1}{1-\beta_1}$；Adam+动量的 Surge 条件 $\beta_1>\frac13$。
- **10542**（2024）[^k10542] — $\eta\propto\sqrt B$（方差守恒）/$\eta\propto B$（SDE）/$\eta^*(B)$ 饱和三推导并列；底标度两口径的源头。
- **10563**（2024）[^k10563] — $\mathrm{softsign}(x,\epsilon)$ 插值：$|g_i|\ll\epsilon$ 的低 SNR 分量退回线性动力学（$\lambda_{\rm crit}$ 以下模式的机制刻画）。
- **11833**（2026）[^k11833] — $F_{\rm opt}\sim\alpha_1(T\eta)^{-\gamma_1}+\alpha_2B^{-\gamma_2}+\alpha_3\eta^{\gamma_3}$（SignSGD/Muon 型 $\gamma_1=\gamma_3=1,\gamma_2=1/2$）；异幂不等式闭式；"指数 $=$ 普适类、系数 $=$ 工程水平"。对照[^bahri21][^mrs22][^kaplan20]。
- **9607**（2023）[^k9607] — $p_k=k^{-\gamma-1}/Z_\gamma\Rightarrow L\sim N^{-\gamma},D^{-\gamma/(\gamma+1)}$；$p_k<\tau/D$ 不可学（§2.5 的输入假设）。
- **11267**（2025）[^k11267] — $\Vert u_t\Vert_{\rm RMS}\approx\sqrt{\frac{\mathrm{SNR}+\kappa}{\mathrm{SNR}+1}}$（§2.6）；同结果见 rotational equilibrium[^kosson23]。
- **11307/11404**（2025）[^k11307][^k11404] — $\Vert\theta\Vert_{\rm RMS}\approx\sqrt{\eta/2\lambda_{\rm wd}}$；动态版 $\frac{d\rho}{dt}\approx-2\lambda_t\eta_t\rho_t+\eta_t^2$；终点 lr$\to0\Rightarrow$ 权重坍缩。
- **11459**（2025）[^k11459] — 优化器只记得最近 $\mathcal O(1/\lambda\eta)$ 步；等权最优条件 $\lambda_s\eta_s+\dot\eta_s/\eta_s\approx0$。
- **11593**（2026）[^k11593] — 恒等式 $\hat v_t=\hat m_t^2+\hat\sigma_t^2$；$u_t=\frac{\mathrm{sign}(\hat m_t)}{\sqrt{1+\hat\sigma_t^2/\hat m_t^2}}$；极小极大证明 $\beta_2=\beta_1$。
- **10588**（2024）[^k10588] — 稳态 $\mathbb E[gg^{\top}]\approx\sigma^2\mathcal H^2\Rightarrow\mathcal H\approx\frac1\sigma\mathrm{diag}(\sqrt{\mathbb E[g\odot g]})$；瞬时（GN/Fisher）与长期平均差一个平方根。

### A.3 Muon 族与谱计算

- **10592**（2024）[^k10592] — msign 恒等式与最优正交近似；谱范数对偶 $\mathrm{tr}(G^{\top}\Phi)\le\Vert G\Vert_\ast$；Shampoo 极限等价 $(GG^{\top})^{-1/4}G(G^{\top}G)^{-1/4}$。
- **10648**（2024）[^k10648] — $\nabla_W\Vert W\Vert_2=u_1v_1^{\top}$；谱 WD 只压 $\sigma_1$ vs F 范数 WD 全谱。
- **10739**（2025）[^k10739] — 最小作用量导出 Muon；WD 界 $\Vert W_t\Vert_2\le\max(\Vert W_0\Vert_2,1/\lambda)$；Update RMS 对齐；奇异值熵。
- **11416**（2025）[^k11416] — 四版本缩放因子表（朴素/Keller/MuP/Moonlight）；消融时必须固定版本。
- **11549**（2026）[^k11549] — 特征层面最速下降需要 $XX^{\top}\approx I$；Muon 下参数级 $\approx$ 特征级。
- **11875**（2026）[^k11875] — $\Phi^*=(X^{\top}X+\lambda I)^{-1}X^{\top}G_Y$：阻尼 Fisher 逆（§2.4 的内插旋钮）。
- **11126**（2025）[^k11126] — $\vert q\cdot k\vert\le\Vert x_i\Vert\Vert x_j\Vert\Vert W_q\Vert_2\Vert W_k\Vert_2$；满秩更新谱 vs 低秩的"碰撞"机制（论文[^liu25]）。
- **10795/11772**（2025–2026）[^k10795][^k11772] — 谱条件缩放 $\Vert W_k\Vert_2=\Theta(\sqrt{d_k/d_{k-1}})$[^yang23spectral]；$\alpha=\sqrt{\max(1,d_{\rm out}/d_{\rm in})}$ 的逐 token RMS 精确推导。
- **10847/11767**（2025–2026）[^k10847][^k11767] — Stable rank $\Vert M\Vert_F^2/\Vert M\Vert_2^2$（Roy & Vetterli[^roy07]）；$q{=}2$ 熵参考点 $\log(n/2)$。
- **11784**（2026）[^k11784] — 投影算子 $\mathcal P_m$ 的闭式（度量学习，无谱内容）。
- **10922/10996**（2025）[^k10922][^k10996] — 标量迭代 $x_{t+1}=ax_t+bx_t^3+cx_t^5$；**Polar Express 贪心 $=$ 全局最优定理**[^amsel25]；小谱初值下限 $0.1/\sqrt{\min(n,m)}$。
- **11006/11025/11059**（2025）[^k11006][^k11025][^k11059] — $\mathrm{mclip}(M)=U\min(\Sigma,1)V^{\top}$ 的 2–3 次 msign 闭式；Sylvester/$\mcsgn$ 反传；误差抵消的奇函数版。
- **11335/11736**（2025–2026）[^k11335][^k11736] — $\mathbb E\Vert W\Vert_2\approx\sqrt n+\sqrt m$（[^davidson01]）；幂迭代 $(\sigma_2/\sigma_1)^{2t}$；Krylov 加速；Schatten 矩上界。
- **11654–11719**（2026）[^k11654][^k11673][^k11697][^k11710][^k11719] — $V_t=\QR(M_t^{\top}M_tV_{t-1})$ 在线子空间跟踪；**SCQR 误差不累积定理**；免费 $\Sigma_t$ 与谱范数读数；近简并谱是唯一失效点；Spectral Hammer（rank-1 流式裁剪）。
- **11196–11241**（2025）[^k11196][^k11215][^k11221][^k11241] — 最小作用量框架；**谱范数球定理**（§2.4）；Stiefel 的连续 Lyapunov 方程解；谱球面约束。

### A.4 动力学、ODE 与统一视角

- **5655**（2018）[^k5655] — SGD $\approx$ 朗之万（平稳分布 $e^{-L/\sigma^2}$，[^smith18bayes]）；动量 $=$ 蛙跳二阶 ODE[^sutskever13]；lr–$\beta$ 耦合 $\beta\to1-(1-\beta)\sqrt r$。
- **6234/6261**（2019）[^k6234][^k6261] — 截断误差控制导出自适应 lr；RMSProp 连续动力学**无不动点**；Polyak 步长 $L(t)=e^{-t}$ 但奇异。
- **6583**（2019）[^k6583] — Dirac GAN 线性化 $\ddot\theta\approx-\theta/4$；权重 EMA 消振荡还原均衡点（对照零中心梯度惩罚的收敛性[^mescheder18]）。
- **7980**（2021）[^k7980] — stop-gradient 的快/慢动力学分解（对照 DirectPred[^tian21directpred]；SimSiam[^chen21simsiam]）。
- **8009**（2021）[^k8009] — $\frac{df_\theta(x)}{dt}=\sum_i\alpha_{\theta,i}K_\theta(x,x_i)$：梯度下降的解恒为核机器（NTK[^jacot18]；细粒度分析[^arora19]）——谱吸收的函数空间版本。
- **9660**（2023）[^k9660] — KL 正则化近端点 $\Rightarrow$ 自然梯度（KL 二阶展开 $=$ Fisher 的连续时间出处）；Wasserstein 梯度流（JKO[^jko98]）$\to$ Fokker–Planck $\to$ SDE。
- **9359**（2022）[^k9359] — 离散传播子 $(I+\Delta xA)^n\approx e^{n\Delta xA}$：与 $(1-\eta\lambda_i)^t$ 同一半群结构（原文只到静态拉普拉斯方程；热核类比是本轮的合法移植）。
- **7521**（2020）[^k7521] — $p(\Delta\theta)\propto e^{-\Delta l/\alpha}$：一阶展开 $\to$ 最速下降、二阶展开 $\to$ 牛顿；温度 $\alpha\approx\eta\sigma^2$。
- **7737**（2020）[^k7737] — 最优 baseline；零阶梯度 $\approx$ 策略梯度的等价定理；方差工程见[^greensmith04]。
- **4277/5239**（2017–2018）[^k4277][^k5239] — GD/EM $=$"局部近似 $+$ 精确极小"家族；$\mathrm{KL}(\tilde p(X,Y)\|p_\theta)=\mathbb E_X[\mathrm{KL}(\tilde p(Y|X)\|p_\theta)]+\text{const}$：OPD 的 $J$ 正是此条件 KL 结构，冻结状态集 $=$ 固定数据坐标。
- **5617**（2018）[^k5617] — $\langle u_i,v_j\rangle=\log\frac{\tilde p(w_j|w_i)}{\tilde p(w_j)}$（PMI 恒等式；对照[^levy14][^mikolov13]）——log 比率 $=$ 打分（与 advantage 同构）。
- **8968**（2022）[^k8968] — $\nu_{t+1}=\nu_t\odot\exp(\gamma g_{t-1}\odot g_t)$：逐坐标梯度自相关读数。

### A.5 蒸馏、扩散与分布匹配

- **9119/9152/9164**（2022）[^k9119][^k9152][^k9164] — 多噪声源合并的积分技巧；$\alpha_t^2+\beta_t^2=1$ 分解；贝叶斯反条件分布（DDPM[^ho20]）。
- **9181**（2022）[^k9181] — 只凭端点边际待定系数：中间路径自由（DDIM[^song21ddim]）——一步拟合端点分布的合法性原型。
- **9209/9228/9280**（2022）[^k9209][^k9228][^k9280] — 反向 SDE 与概率流 ODE（Score-SDE[^song21sde]）；同边际对应一族 $\sigma_t$ 过程：路径重参数化是自由旋钮。
- **9467**（2023）[^k9467] — $\mathcal W_2\le\int g_t^2I_t\big(\mathbb E\Vert s-\nabla\log p\Vert^2\big)^{1/2}dt+\cdots$：代理目标 $\le$ 分布距离的定量桥。
- **9509**（2023）[^k9509] — 差 $\theta$-无关 $\Rightarrow$ 同最优解（Vincent 2011[^vincent11] 的扩散版）。
- **9662/9668**（2024）[^k9662][^k9668] — 把样本空间一步位移压进参数更新；ReFlow 损失 $=$ WGAN-GP 判别器损失（Rectified Flow[^liu22rf]；Flow Matching[^lipman22]；PFGM[^xu22pfgm] 见 9305）。
- **9881**（2024）[^k9881] — 一步误差 $\le$ 逐段之和；轨迹 top-1/2 主成分承载精度：**少步误差集中在极低维子空间**。
- **10047/10055**（2024）[^k10047][^k10055] — $\mathrm{SNR}^{\text{高}}\times s^2=\mathrm{SNR}^{\text{低}}$ 对齐公式：可分辨性阈值的显式迁移（$\lambda_{\rm crit}$ 同构；EDM 的 SNR 预条件[^karras22]）。
- **10077**（2024）[^k10077] — 加速降低采样映射非线性容量（Shortcut[^frans24]）。
- **10085/10567**（2024–2025）[^k10085][^k10567] — SiD[^zhou24sid]：$\mathbb E[\langle f,\epsilon_{\varphi^*}\rangle]=\mathbb E[\langle f,\varepsilon\rangle]$；$\mathcal L_4^{\rm(sg)}=2\mathcal L_2^{\rm(sg)}-\mathcal L_1^{\rm(sg)}$ 梯度恒等；广义 Fisher 散度坍缩分析。
- **10617/10633**（2025）[^k10617][^k10633] — 自洽约束 $v(x_t,t,2\epsilon)=\frac12[v(x_t,t,\epsilon)+v(\tilde x,t+\epsilon,\epsilon)]$；CT 目标 $+$ EMA 超前解（[^song23cm][^lu24scm]）。
- **10958**（2025）[^k10958] — 平均速度恒等式 $v=u+(t-r)\frac d{dt}u$；sg 的驻点警告（[^geng25meanflow]）。
- **11428**（2026）[^k11428] — 低秩瓶颈下只有 $x$-pred 可训：预测目标重参数化 $=$ 谱过滤（对照 v-参数化[^salimans22]）。
- **4439/6016/6051/6280/8244**（2017–2021）[^k4439][^k6016][^k6051][^k6280][^k8244] — WGAN-GP 从稳定性公理导出[^gulrajani17]；f-GAN 凸共轭表[^nowozin16]；Lipschitz/谱归一化[^miyato18]；KR 对偶[^villani08]；"WGAN 成功与 W 距离无关"[^fedus18]。
- **5716/6316/6331/6612**（2018–2019）[^k5716][^k6316][^k6331][^k6612] — **$\mathrm{KL}(q_\theta\|q_{\theta-\Delta\theta})\approx\frac12\Delta\theta^{\top}F\Delta\theta$（Fisher 信赖域）**；缺完备项 $\Rightarrow$ mode collapse；Langevin 负相 $=$ reverse-KL 最小化（EBM[^du19ebm]）。
- **6705**（2019）[^k6705] — SF/REINFORCE vs 重参数方差；Gumbel-Max 精确性定理（[^jang17][^maddison17]）。
- **10226/10266/10001**（2024）[^k10226][^k10266][^k10001] — LoRA 首步更新 $=$ 初始梯度 top-2r 奇异子空间（Eckart–Young）（[^wang24loraga][^wang24lorapro]；LoRA[^hu21lora]）；$\eta_B/\eta_A\approx\mathcal O(\sqrt n)$[^hayou24]。
- **10489**（2024）[^k10489] — STE 一般化 $\partial q/\partial z=G$ 的尺度失配警告（STE[^bengio13ste]；VQ-VAE[^oord17]）。
- **9370/9379/9497/9305/10711**（2022–2025）[^k9370][^k9379][^k9497][^k9305][^k10711] — 特征线法、边际场 $=$ 条件场后验期望、格林函数构造。
- **11664/10017/10373**（2024–2026）[^k11664][^k10017][^k10373] — 残差 $=$ 全历史等权和（Hyper-Connections[^zhu24hyper]；RealFormer[^he20realformer]）；"新聚合须覆盖旧聚合特例"原则。

### A.6 换目标 $=$ 换谱（loss 设计与 softmax 几何）

- **10289**（2024）[^k10289] — 单纯形一阶方向恒为顶点 onehot（Frank–Wolfe 型）；连续情形 $=$ Wasserstein 梯度流。
- **8512**（2021）[^k8512] — KL/巴氏/$\mathcal W_2$ 闭式（§2.7 的静态对照）。
- **9039**（2022）[^k9039] — $D(p,q)=\langle f(s)-f(t),s-t\rangle$：KL 读数 $=$ Fisher 度量显式形；结构化输出的口径警告。
- **11854**（2026）[^k11854] — 可采样性 $\Rightarrow$ $S=H+(\boldsymbol e_i-\boldsymbol q)\cdot\nabla H$（Fenchel–Young）；**只依赖 $q_i$ 的评分只有对数评分**；Brier 梯度 $=2(\mathrm{diag}(q)-qq^{\top})(q-e_i)$；reverse-KL 的 score 依赖 $p$ 本身（不满足线性公理）——OPD 被迫用 CE 型代理的结构原因。
- **10145/11814**（2024–2026）[^k10145][^k11814] — $\frac12\Vert\partial p/\partial x\Vert_1=1-\sum p_i^2$（Sparsemax[^martins16]、Entmax[^peters19]）；softmax 泰勒展开：二阶项 $=$ Fisher、三/四阶项系数 $=$ logit 中心矩（A2 残差的显式阶）。
- **9526/9098/9797/4733**（2017–2023）[^k9526][^k9098][^k9797][^k4733] — TaiLr 有理权（压 $1/p$ 奇点）；CE 后期 $\approx$ 准确率口径（$\log x\approx x-1$）；EMO 最优传输加权；focal $=$ 硬截断的光滑化[^lin17focal]。
- **8823/9034/9588/9812**（2021–2023）[^k8823][^k9034][^k9588][^k9812] — $H\approx\log n-\lambda(s_{\max}-\bar s)\Rightarrow\lambda\propto\log n$；$\alpha^*\approx0.84\sqrt{\log n}$（建议 $2.5/\sqrt d$）：**温度同乘 Jacobian $=$ 纯时间重标定候选**。

### A.7 信息论与谱的起源

- **5448/5476/5577/6191/7695**（2018–2020）[^k5448][^k5476][^k5577][^k6191][^k7695] — 学习 $=$ 熵减（MDL 传统[^rissanen78]）；合并增益 $\mathcal F^*_{ab}=p_{ab}\big(\ln\frac{p_{ab}}{p_ap_b}-1\big)$（信噪增益 $-$ 固定成本，与 $\lambda_{\rm crit}$ 逐项对应）；$n>8.33\log N$ 容量线；最小熵 $=$ 最大似然。
- **9595**（2023）[^k9595] — $S^*_{1,2}=\frac{\mathbb E|x|}{\sqrt{\mathbb E[x^2]}}$；公理体系（[^hurley09]）。
- **8679/8706**（2021）[^k8679][^k8706] — $n>\frac{24\log N}{\varepsilon^2}$[^dasgupta03]；随机 SVD（[^halko11]）。
- **3534/6181**（2015–2018）[^k3534][^k6181] — 函数方程 $\Rightarrow S=-\sum p\log p$；VIB 逐维 KL 解耦（[^alemi17]）。
- **8578/6747**（2019–2021）[^k8578][^k6747] — 同协方差高斯 $\Rightarrow$ 闭式 $\boldsymbol w=(\mu_1-\mu_0)\Sigma^{-1}$（少样本下闭式二阶矩 $>$ 一阶迭代）；$\mathbb E[\hat\sigma^2_{1/B}]=\frac{B-1}B\sigma^2$（方差估计的偏差修正）。

### A.8 参数化、初始化与深度稳定性

- **10770**（2025）[^k10770] — SGD/Adam/Muon 的方差-lr 缩放表；lr 尺度律由更新范数（F/L1/核范数）决定（μP[^yang22mup]）。
- **11340/11605/11647/11729**（2025–2026）[^k11340][^k11605][^k11647][^k11729] — 好模型三特征（前向/依赖/更新均 $\Theta(1)$）；线性层谱约束最速下降；**LM head 几何**（$\vert\log\sum e^{a_i}-\log\sum e^{b_i}\vert\le\max_i\vert a_i-b_i\vert\Rightarrow$ 输出层逐列 normalized SGD、$\sigma=\Theta(1/d)$）；Pre Decay $=$ 只保上界的谱裁剪。
- **8620/8747/7180**（2020–2021）[^k8620][^k8747][^k7180] — 二阶矩记账（He init[^he15]；Glorot[^glorot10]；正交初始化[^saxe13]）；Adam 常量更新 $\mathcal O(\eta)$ 与 Post-Norm 的 0.02；随机初始化 $\approx$ 正交 $\Rightarrow$ 初始振幅近各向同性。
- **6992/10831**（2019–2025）[^k6992][^k10831] — 降 Lipschitz 常数 $L$（BN[^ioffe15]；平滑性视角[^santurkar18]）；RMSNorm[^zhang19rms] Jacobian $=\frac1{\Vert x\Vert}\big(I-\frac{yy^{\top}}d\big)$；DyT 的 ODE 推导（LayerNorm[^ba16]；DyT[^zhu25dyt]）。
- **8978/8994/9009/8444/11371**（2021–2025）[^k8978][^k8994][^k9009][^k8444][^k11371] — 增量爆炸 $\Delta L=\mathcal O(\eta NK)$ 与优化器相关补偿指数（DeepNet[^yang22deepnet]）；残差 $\varepsilon=1/\sqrt N$（高维近正交把 $(1+\varepsilon)^N$ 降为 $(1+\varepsilon^2)^{N/2}$）；Pre Norm $=$ 加宽非加深（[^xiong20]）；无损放大的逐层尺度账本；低精度注意力的有偏舍入。
- **8069/9079**（2021–2022）[^k8069][^k9079] — $\tilde x=(x-\mu)U\sqrt{\Lambda^{-1}}$，可递归在线计算（BERT-flow[^li20flow] 的线性替代）；**白化的向量空间先例**。
- **9698/9762**（2023）[^k9698][^k9762] — 共享 embedding 初始损失 $\approx d\sigma\gg\log n$，修正 $\sigma=(\log n)/d$（[^press17][^li21coupling]）。
- **7888/7469/10657/8796**（2020–2025）[^k7888][^k7469][^k10657][^k8796] — RNN 梯度连乘谱（LSTM[^hochreiter97]；梯度消失[^bengio94]）；$(L_0,L_1)$-smooth 下裁剪 $=$ 自适应步长（梯度裁剪[^pascanu13clip]）；"默认模长 1" $=$ 平稳梯度模长；输入/参数梯度惩罚不等式。
- **6794/8634/6869/7094**（2019–2021）[^k6794][^k8634][^k6869][^k7094] — 累积 $=$ 大 batch；动量 $\equiv$ 累积的 $\tilde\beta=1-\frac{1-\beta}k$ 换算；Lookahead[^zhang19lookahead]；LAMB 型层自适应[^you20lamb]。
- **7302/7387**（2020）[^k7302][^k7387] — rank-1 二阶矩（广义 KL 的解析解 $=$ 行列边缘，Adafactor[^shazeer18]）；$\hat\beta_{2,t}=1-t^{-c}$ 退火（AdamW[^loshchilov19]）。
- **9473/9512**（2023）[^k9473][^k9512] — sign 更新 $=$ 盲谱均匀化极限（每步注入 $\sim\eta\Vert g\Vert_1$ 噪声）[^chen23lion]；$\gamma_t$ 插值 Tiger$\to$SGDM。
- **7196/9344/9590/8870/8896/8907**（2022–2023）[^k7196][^k9344][^k9590][^k8870][^k8896][^k8907] — 逐块 $\lambda_{\rm crit}$ 的实证；Amos 的 $\rho\propto\alpha^2$ 自洽与 $p_t\propto e^{-S_t}$ 衰减[^tian22amos]；LoRA $=$ 秩 $r$ 投影组合；stop-gradient 加权恒等式与几何平均；MGDA 最小范数聚合（[^sener18]；GradNorm[^chen18gradnorm]）。
- **10699–11782**（MoE 环游记，2025–2026）[^k10699][^k10815][^k11619][^k11760][^k11782] — 正交假设下按模长 Top-k 最优；STE 恒等式 $\sum_i\nabla P_i=0$；整数规划对偶解 $=$ 分位数（Loss-Free[^wang24lossfree]；Switch[^fedus22]；Expert-Choice[^zhou22ec]；稀疏 MoE[^shazeer17]）。
- **11320/11486/11823**（2025–2026）[^k11320][^k11486][^k11823] — EFLA：精确流谱恒在 $(0,1]$ 而欧拉离散可爆炸；Softmax Attention 的 Delta-rule 恒等式（[^yang24deltanet][^yang24gated]）。
- **10114/10162/9783/8601/10091/10907**（2021–2025）[^k10114][^k10162][^k9783][^k8601][^k10091][^k10907] — HiPPO/S4 的最优多项式投影（[^gu20hippo][^gu21s4]）；MLA 的 KV 压缩。
- **8128/7259/7500/9617/7430/9019/9889**（2020–2023）[^k8128][^k7259][^k7500][^k9617][^k7430][^k9019][^k9889] — 重复率 $R=\sum_i\frac{\lambda_i(P\odot P)}{1-\lambda_i(P\odot P)}$（转移谱 $\to1$ 的熵坍缩）；Exposure Bias 的局部/全局目标缺口（scheduled sampling[^bengio15ss]）；NBCE；熵不变 scale。

### A.9 其余低相关（已提取，仅列）

- 词向量与句向量：4299[^k4299]、4667/4675[^k4667][^k4675]（word2vec/GloVe 式目标推导，[^mikolov13]）、8656[^k8656]（Margin Softmax）、9368[^k9368]（测地线）、11818[^k11818]（排序不等式）、7388[^k7388]（EMD/WMD）、9336[^k9336]（CUR）、8671[^k8671]（NSP zero-shot）、9046[^k9046]（无法预测的词）。
- 结构化 IE：8373[^k8373]、8877[^k8877]（GlobalPointer 及其高效版）、4695[^k4695]（CRF）、7213[^k7213]（MEMM）。

---

## 附录 B：覆盖与排除

**提取批次（15 批，$\approx$210 篇精读/略读）**：Muon 核心（12）；msign/谱工具/流形（16）；炼丹系列（10）；lr/bs/缩放律（14）；动力学/ODE（14）；优化器家族（16）；MuP/参数化（10）；扩散/蒸馏（18）；GAN/f-散度（10）；loss/KL/最优分布（14）；信息论/最小熵（12）；架构/MoE/解码略读（24）；查漏补缺（18）；whitening/输出端（主 agent 直读 4 篇）；其余词向量/CRF 类按标题略读排除。

**排除口径（$\approx$300 篇）**：竞赛/爬虫/工具/代码分享/数据资源/模型发布类无数学推导的文章；注意力位置编码系列（与优化动力学无直接交集）；MoE 应用篇。如需补扫，按 ID 查 `kexue/articles/<ID>.md`。

**勘误**：10289/8512/8578/6747 提取时个别 agent 报本地缺档，已从 kexue.fm 原文核对（内容已入总库）；11563（DeltaNet 逆矩阵元素界）未入批，属低相关。

---

### 参考文献

[^fu26]: Z. Fu, B. He, Y. Zuo, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^opd1-26]: Z. Fu, et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe (Part I). arXiv 2026. [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)

[^cai26]: Y. Cai, D. Cao, L. Lin, et al. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^gu24]: Y. Gu, L. Dong, F. Wei, M. Huang. MiniLLM: Knowledge Distillation of Large Language Models. ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)

[^agarwal24]: R. Agarwal, N. Vieillard, Y. Zhou, et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD). ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

[^ross11]: S. Ross, G. Gordon, J. A. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)

[^zinkevich03]: M. Zinkevich. Online Convex Programming and Generalized Infinitesimal Gradient Ascent. ICML 2003.

[^greensmith04]: E. Greensmith, P. L. Bartlett, J. Baxter. Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. JMLR 5:1471–1530, 2004.

[^kakade02]: S. Kakade. A Natural Policy Gradient. NeurIPS 2002.

[^schulman15]: J. Schulman, S. Levine, P. Moritz, M. I. Jordan, P. Abbeel. Trust Region Policy Optimization. ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)

[^schulman17]: J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal Policy Optimization Algorithms. arXiv 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^tomar21]: M. Tomar, U. A. Mishra, A. Rajeswaran, L. Yang. Mirror Descent Policy Optimization. AAAI 2021.

[^amari98]: S. Amari. Natural Gradient Works Efficiently in Learning. Neural Computation 10(2):251–276, 1998.

[^martens14]: J. Martens. New Insights and Perspectives on the Natural Gradient Method. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^martens-grosse-15]: J. Martens, R. Grosse. Optimizing Neural Networks with Kronecker-factored Approximate Curvature. ICML 2015. [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)

[^gupta18]: V. Gupta, T. Koren, Y. Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. ICML 2018. [arXiv:1802.09568](https://arxiv.org/abs/1802.09568)

[^vyas24]: N. Vyas, D. Morwani, R. Zhao, et al. SOAP: Improving and Stabilizing Shampoo using Adam for Deep LLM Training. arXiv 2024. [arXiv:2409.11321](https://arxiv.org/abs/2409.11321)

[^li18psgd]: X. Li. Preconditioned Stochastic Gradient Descent. IEEE Trans. Neural Netw. Learn. Syst. 2018. [arXiv:1512.04202](https://arxiv.org/abs/1512.04202)

[^agarwal12]: A. Agarwal, P. L. Bartlett, P. Ravikumar, M. J. Wainwright. Information-Theoretic Lower Bounds on the Oracle Complexity of Stochastic Convex Optimization. IEEE Trans. Inf. Theory 58(5):3235–3249, 2012.

[^bach-moulines-13]: F. Bach, E. Moulines. Non-asymptotic Analysis of Stochastic Approximation Algorithms for Machine Learning. NeurIPS 2013.

[^dieuleveut-17]: A. Dieuleveut, N. Flammarion, F. Bach. Harder, Better, Faster, Stronger Convergence Rates for Least-Squares Regression. JMLR 18(101):1–52, 2017.

[^dieuleveut16]: A. Dieuleveut, F. Bach. Nonparametric Stochastic Approximation with Large Step-Sizes. Annals of Statistics 44(4):1363–1399, 2016.

[^yao07]: Y. Yao, L. Rosasco, A. Caponnetto. On Early Stopping in Gradient Descent Learning. Constructive Approximation 26:289–315, 2007.

[^mandt17]: S. Mandt, M. Hoffman, D. Blei. Stochastic Gradient Descent as Approximate Bayesian Inference. JMLR 18(134):1–35, 2017. [arXiv:1704.04289](https://arxiv.org/abs/1704.04289)

[^uhlenbeck30]: G. E. Uhlenbeck, L. S. Ornstein. On the Theory of the Brownian Motion. Physical Review 36:823–841, 1930.

[^li17sme]: Q. Li, C. Tai, W. E. Stochastic Modified Equations and Dynamics of Stochastic Gradient Algorithms I: Non-Convex Learning. arXiv 2017. [arXiv:1705.10405](https://arxiv.org/abs/1705.10405)

[^smith18bayes]: S. L. Smith, Q. V. Le. A Bayesian Perspective on Generalization and Stochastic Gradient Descent. ICLR 2018. [arXiv:1710.06451](https://arxiv.org/abs/1710.06451)

[^smith18decay]: S. L. Smith, P.-J. Kindermans, C. Ying, Q. V. Le. Don't Decay the Learning Rate, Increase the Batch Size. ICLR 2018. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489)

[^mccandlish18]: S. McCandlish, J. Kaplan, D. Amodei, et al. An Empirical Model of Large-Batch Training. arXiv 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

[^barrett20]: D. G. Barrett, B. G. Dherin. Implicit Gradient Regularization. ICLR 2021. [arXiv:2009.11162](https://arxiv.org/abs/2009.11162)

[^cohen21]: J. Cohen, A. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar. Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. ICLR 2021. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)

[^feller71]: W. Feller. An Introduction to Probability Theory and Its Applications, Vol. II. Wiley, 1971.（Karamata–Tauberian 定理与完全单调性）

[^proschan63]: F. Proschan. Theoretical Explanation of Observed Decreasing Failure Rate. Technometrics 5(3):375–383, 1963.

[^palmer84]: R. G. Palmer, D. Stein, E. Abrahams, P. W. Anderson. Models of Hierarchically Constrained Relaxation for Glassy Relaxation. Phys. Rev. Lett. 53:958–961, 1984.

[^kingma15]: D. Kingma, J. Ba. Adam: A Method for Stochastic Optimization. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

[^loshchilov19]: I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

[^duchi11]: J. Duchi, E. Hazan, Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR 12:2121–2159, 2011.

[^balles18]: L. Balles, P. Hennig. Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients. ICML 2018. [arXiv:1705.07774](https://arxiv.org/abs/1705.07774)

[^bernstein18]: J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, A. Anandkumar. signSGD: Compressed Optimisation for Non-Convex Problems. ICML 2018. [arXiv:1802.04434](https://arxiv.org/abs/1802.04434)

[^reddi18]: S. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. ICLR 2018. [arXiv:1710.05965](https://arxiv.org/abs/1710.05965)

[^kosson23]: C. Kosson, et al. Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks. NeurIPS 2023. [arXiv:2305.17212](https://arxiv.org/abs/2305.17212)

[^li24surge]: S. Li, et al. Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling. NeurIPS 2024. [arXiv:2405.14578](https://arxiv.org/abs/2405.14578)

[^tian22amos]: R. Tian, et al. Amos: An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale. arXiv 2022. [arXiv:2210.11693](https://arxiv.org/abs/2210.11693)

[^chen23lion]: X. Chen, C. Liang, et al. Symbolic Discovery of Optimization Algorithms (Lion). NeurIPS 2023. [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)

[^shazeer18]: N. Shazeer, M. Stern. Adafactor: Adaptive Learning Rates with Sublinear Memory. ICML 2018. [arXiv:1804.04235](https://arxiv.org/abs/1804.04235)

[^zhang19lookahead]: M. R. Zhang, J. Lucas, J. Ba, G. Hinton. Lookahead Optimizer: k Steps Forward, 1 Step Back. NeurIPS 2019. [arXiv:1907.08610](https://arxiv.org/abs/1907.08610)

[^you20lamb]: Y. You, et al. Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes (LAMB). ICLR 2020. [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)

[^defazio23]: A. Defazio, X. A. Yang, H. Mehta, K. Mishchenko, A. Khaled, A. C. Wilson. Optimal Linear Decay Learning Rate Schedules and Further Refinements. arXiv 2023. [arXiv:2310.07831](https://arxiv.org/abs/2310.07831)

[^defazio24]: A. Defazio, X. A. Yang, H. Mehta, K. Mishchenko, A. Khaled, A. C. Wilson. The Road Less Scheduled (Schedule-Free). arXiv 2024. [arXiv:2405.15682](https://arxiv.org/abs/2405.15682)

[^polyak92]: B. T. Polyak, A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM J. Control Optim. 30(4):838–855, 1992.

[^izmailov18]: D. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson. Averaging Weights Leads to Wider Optima and Better Generalization (SWA). UAI 2018. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)

[^sutskever13]: I. Sutskever, J. Martens, G. Dahl, G. Hinton. On the Importance of Initialization and Momentum in Deep Learning. ICML 2013.

[^goh17]: C. Olah. Why Momentum Really Works. Distill 2017. [https://distill.pub/2017/momentum/](https://distill.pub/2017/momentum/)

[^jordan24]: K. Jordan, et al. Muon: Momentum Orthogonalized by Newton–Schulz. 开源优化器 2024. [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)

[^liu25]: J. Liu, J. Su, X. Yao, et al. Muon is Scalable for LLM Training. arXiv 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

[^amsel25]: N. Amsel, J. Persson, C. Musco, R. M. Gower, et al. The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm. arXiv 2025. [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)

[^higham08]: N. J. Higham. Functions of Matrices: Theory and Computation. SIAM, 2008.

[^roy07]: O. Roy, M. Vetterli. The Effective Rank: A Measure of Effective Dimensionality. EUSIPCO 2007.

[^davidson01]: K. R. Davidson, S. J. Szarek. Local Operator Theory, Random Matrices and Banach Spaces. In: Handbook of the Geometry of Banach Spaces, Vol. 1. Elsevier, 2001.

[^halko11]: N. Halko, P.-G. Martinsson, J. A. Tropp. Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions. SIAM Review 53(2):217–288, 2011.

[^dasgupta03]: S. Dasgupta, A. Gupta. An Elementary Proof of a Theorem of Johnson and Lindenstrauss. Random Structures & Algorithms 22(1):60–65, 2003.

[^hurley09]: N. Hurley, S. Rickard. Comparing Measures of Sparsity. IEEE Trans. Inf. Theory 55(10):4723–4741, 2009.

[^yang22mup]: G. Yang, E. J. Hu, et al. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. arXiv 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)

[^yang23spectral]: G. Yang, J. B. Simon, J. Bernstein. A Spectral Condition for Feature Learning. arXiv 2023. [arXiv:2310.17813](https://arxiv.org/abs/2310.17813)

[^michaud23]: E. J. Michaud, Z. Liu, U. Girit, M. Tegmark. The Quantization Model of Neural Scaling. NeurIPS 2023. [arXiv:2303.13506](https://arxiv.org/abs/2303.13506)

[^kaplan20]: J. Kaplan, S. McCandlish, T. Henighan, et al. Scaling Laws for Neural Language Models. arXiv 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

[^bahri21]: Y. Bahri, E. Dyer, J. Kaplan, J. Lee, U. Sharma. Explaining Neural Scaling Laws. PNAS 2024. [arXiv:2102.06701](https://arxiv.org/abs/2102.06701)

[^mrs22]: A. Maloney, D. A. Roberts, J. Sully. A Solvable Model of Neural Scaling Laws. arXiv 2022. [arXiv:2210.16859](https://arxiv.org/abs/2210.16859)

[^jacot18]: A. Jacot, F. Gabriel, C. Hongler. Neural Tangent Kernel: Convergence and Generalization in Neural Networks. NeurIPS 2018. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)

[^arora19]: S. Arora, S. S. Du, W. Hu, Z. Li, R. Salakhutdinov, R. Wang. Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks. ICML 2019. [arXiv:1901.08584](https://arxiv.org/abs/1901.08584)

[^kunstner19]: F. Kunstner, P. Hennig, L. Balles. Limitations of the Empirical Fisher Approximation for Natural Gradient Descent. NeurIPS 2019. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)

[^basri20]: R. Basri, D. Jacobs, Y. Kasten, S. Kritchman. The Spectrum of the Neural Tangent Kernel at Initialization. arXiv 2019. [arXiv:1907.08679](https://arxiv.org/abs/1907.08679)

[^rahaman19]: N. Rahaman, A. Baratin, D. Arpit, et al. On the Spectral Bias of Neural Networks. ICML 2019. [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)

[^xu20]: Z. Q. J. Xu, Y. Zhang, T. Luo. Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks. Communications in Computational Physics 28(5):1746–1767, 2020. [arXiv:1901.06523](https://arxiv.org/abs/1901.06523)

[^bordelon20]: B. Bordelon, A. Canatar, C. Pehlevan. Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks. ICML 2020. [arXiv:2002.02561](https://arxiv.org/abs/2002.02561)

[^canatar21]: A. Canatar, B. Bordelon, C. Pehlevan. Spectral Bias and Task-Model Alignment Explain Generalization in Kernel Regression. Nature Communications 12:2914, 2021. [arXiv:2006.13198](https://arxiv.org/abs/2006.13198)

[^yang22deepnet]: H. Wang, et al. DeepNet: Scaling Transformers to 1,000 Layers. EMNLP 2022. [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)

[^xiong20]: R. Xiong, et al. On Layer Normalization in the Transformer Architecture. ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)

[^ba16]: J. L. Ba, J. R. Kiros, G. E. Hinton. Layer Normalization. arXiv 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

[^zhang19rms]: B. Zhang, R. Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

[^ioffe15]: S. Ioffe, C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

[^santurkar18]: S. Santurkar, D. Tsipras, A. Ilyas, A. Madry. How Does Batch Normalization Help Optimization? NeurIPS 2018. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)

[^zhu25dyt]: J. Zhu, X. Chen, K. He, Y. LeCun, Z. Liu. Transformers without Normalization. CVPR 2025. [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)

[^glorot10]: X. Glorot, Y. Bengio. Understanding the Difficulty of Training Deep Feedforward Neural Networks. AISTATS 2010.

[^he15]: K. He, X. Zhang, S. Ren, J. Sun. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. ICCV 2015. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)

[^saxe13]: A. M. Saxe, J. L. McClelland, S. Ganguli. Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks. ICLR 2014. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)

[^press17]: O. Press, L. Wolf. Using the Output Embedding to Improve Language Models. EACL 2017. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)

[^li21coupling]: Z. Li, et al. Rethinking Embedding Coupling in Pre-trained Language Models. ICLR 2021. [arXiv:2010.12821](https://arxiv.org/abs/2010.12821)

[^li20flow]: B. Li, H. Zhou, J. He, et al. On the Sentence Embeddings from Pre-trained Language Models (BERT-flow). EMNLP 2020. [arXiv:2011.05864](https://arxiv.org/abs/2011.05864)

[^tian21directpred]: Y. Tian, X. Chen, S. Ganguli. Understanding Self-Supervised Learning Dynamics without Contrastive Pairs and without Stop-Gradients. NeurIPS 2021. [arXiv:2012.06226](https://arxiv.org/abs/2012.06226)

[^chen21simsiam]: X. Chen, K. He. Exploring Simple Siamese Representation Learning. CVPR 2021. [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)

[^hochreiter97]: S. Hochreiter, J. Schmidhuber. Long Short-Term Memory. Neural Computation 9(8):1735–1780, 1997.

[^bengio94]: Y. Bengio, P. Simard, P. Frasconi. Learning Long-Term Dependencies with Gradient Descent is Difficult. IEEE Trans. Neural Netw. 5(2):157–166, 1994.

[^pascanu13clip]: R. Pascanu, T. Mikolov, Y. Bengio. On the Difficulty of Training Recurrent Neural Networks. ICML 2013. [arXiv:1211.0504](https://arxiv.org/abs/1211.0504)

[^hu21lora]: E. J. Hu, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

[^wang24loraga]: Z. Wang, et al. LoRA-GA: Low-Rank Adaptation with Gradient Approximation. NeurIPS 2024. [arXiv:2407.05000](https://arxiv.org/abs/2407.05000)

[^wang24lorapro]: Z. Wang, et al. LoRA-Pro: Are Low-Rank Adapters Properly Optimized? ICLR 2025. [arXiv:2407.18242](https://arxiv.org/abs/2407.18242)

[^hayou24]: S. Hayou, N. Ghosh, B. Yu. LoRA+: Efficient Low Rank Adaptation with Unequal Learning Rates. arXiv 2024. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)

[^sener18]: O. Sener, V. Koltun. Multi-Task Learning as Multi-Objective Optimization. NeurIPS 2018. [arXiv:1810.04650](https://arxiv.org/abs/1810.04650)

[^chen18gradnorm]: Z. Chen, et al. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML 2018. [arXiv:1711.02257](https://arxiv.org/abs/1711.02257)

[^shazeer17]: N. Shazeer, et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

[^fedus22]: W. Fedus, B. Zoph, N. Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR 2022. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

[^zhou22ec]: Y. Zhou, T. Lei, et al. Mixture-of-Experts with Expert Choice Routing. NeurIPS 2022. [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)

[^wang24lossfree]: L. Wang, H. Gao, C. Zhao, X. Sun, D. Dai. Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv 2024. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)

[^gu20hippo]: A. Gu, T. Dao, S. Ermon, A. Rudra, C. Ré. HiPPO: Recurrent Memory with Optimal Polynomial Projections. NeurIPS 2020. [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)

[^gu21s4]: A. Gu, K. Goel, C. Ré. Efficiently Modeling Long Sequences with Structured State Spaces (S4). ICLR 2022. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)

[^yang24deltanet]: S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 2024. [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)

[^yang24gated]: S. Yang, J. Kautz, A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

[^zhu24hyper]: D. Zhu, et al. Hyper-Connections. ICLR 2025. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)

[^he20realformer]: R. He, et al. RealFormer: Transformer Likes Residual Attention. arXiv 2020. [arXiv:2012.11747](https://arxiv.org/abs/2012.11747)

[^ho20]: J. Ho, A. Jain, P. Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[^song21ddim]: J. Song, C. Meng, S. Ermon. Denoising Diffusion Implicit Models. ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

[^song21sde]: Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

[^karras22]: T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the Design Space of Diffusion-Based Generative Models (EDM). NeurIPS 2022. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)

[^salimans22]: T. Salimans, J. Ho. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

[^lipman22]: Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. Flow Matching for Generative Modeling. ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

[^liu22rf]: X. Liu, C. Gong, Q. Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

[^xu22pfgm]: Y. Xu, Z. Liu, M. Tegmark, T. Jaakkola. PFGM: The Poisson Flow Generative Model. NeurIPS 2022. [arXiv:2209.11178](https://arxiv.org/abs/2209.11178)

[^vincent11]: P. Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation 23(7):1661–1674, 2011.

[^zhou24sid]: M. Zhou, H. Zheng, Z. Wang, M. Yin, H. Huang. Score Identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation. arXiv 2024. [arXiv:2404.04057](https://arxiv.org/abs/2404.04057)

[^geng25meanflow]: Z. Geng, M. Deng, X. Bai, J. Z. Kolter, K. He. Mean Flows for One-step Generative Modeling. NeurIPS 2025. [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)

[^song23cm]: Y. Song, P. Dhariwal, M. Chen, I. Sutskever. Consistency Models. ICML 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)

[^lu24scm]: C. Lu, Y. Song. Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models. arXiv 2024. [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)

[^frans24]: K. Frans, D. Hafner, S. Levine, P. Abbeel. One Step Diffusion via Shortcut Models. ICLR 2025. [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)

[^nowozin16]: S. Nowozin, B. Cseke, R. Tomioka. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. NeurIPS 2016. [arXiv:1606.00709](https://arxiv.org/abs/1606.00709)

[^gulrajani17]: I. Gulrajani, et al. Improved Training of Wasserstein GANs. NeurIPS 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)

[^fedus18]: I. Fedus, M. Rosca, et al. Many Paths to Equilibrium: GANs Do Not Need to Decrease a Divergence At All. ICLR 2018. [arXiv:1710.08446](https://arxiv.org/abs/1710.08446)

[^mescheder18]: L. Mescheder, A. Geiger, S. Nowozin. Which Training Methods for GANs do actually Converge? ICML 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)

[^miyato18]: T. Miyato, et al. Spectral Normalization for Generative Adversarial Networks. ICLR 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)

[^villani08]: C. Villani. Optimal Transport: Old and New. Springer, 2008.

[^jko98]: R. Jordan, D. Kinderlehrer, F. Otto. The Variational Formulation of the Fokker–Planck Equation. SIAM J. Math. Anal. 29(1):1–17, 1998.

[^du19ebm]: Y. Du, I. Mordatch. Implicit Generation and Modeling with Energy-Based Models. NeurIPS 2019. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)

[^jang17]: E. Jang, S. Gu, B. Poole. Categorical Reparameterization with Gumbel-Softmax. ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)

[^maddison17]: C. J. Maddison, A. Mnih, Y. W. Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. ICLR 2017. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)

[^bengio13ste]: Y. Bengio, N. Léonard, A. Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. arXiv 2013. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

[^oord17]: A. van den Oord, O. Vinyals, K. Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

[^lin17focal]: T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. Focal Loss for Dense Object Detection. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

[^martins16]: A. F. T. Martins, R. F. Astudillo. From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. ICML 2016. [arXiv:1602.02068](https://arxiv.org/abs/1602.02068)

[^peters19]: B. Peters, V. Niculae, A. F. T. Martins. Sparse Sequence-to-Sequence Models (Entmax). ACL 2019. [arXiv:1905.07720](https://arxiv.org/abs/1905.07720)

[^alemi17]: A. Alemi, I. Fischer, J. Dillon, K. Murphy. Deep Variational Information Bottleneck. ICLR 2017. [arXiv:1612.00410](https://arxiv.org/abs/1612.00410)

[^rissanen78]: J. Rissanen. Modeling by Shortest Data Description. Automatica 14(5):465–471, 1978.

[^mikolov13]: T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean. Distributed Representations of Words and Phrases and their Compositionality. NeurIPS 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)

[^levy14]: O. Levy, Y. Goldberg. Neural Word Embedding as Implicit Matrix Factorization. NeurIPS 2014.

[^bengio15ss]: S. Bengio, O. Vinyals, N. Jaitly, N. Shazeer. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. NeurIPS 2015. [arXiv:1506.03099](https://arxiv.org/abs/1506.03099)

[^k9902]: 苏剑林. 让炼丹更科学一些（一）：SGD的平均损失收敛. kexue.fm 2023. [https://kexue.fm/archives/9902](https://kexue.fm/archives/9902)

[^k11469]: 苏剑林. 让炼丹更科学一些（二）：将结论推广到无界域. kexue.fm 2025. [https://kexue.fm/archives/11469](https://kexue.fm/archives/11469)

[^k11480]: 苏剑林. 让炼丹更科学一些（三）：SGD的终点损失收敛. kexue.fm 2025. [https://kexue.fm/archives/11480](https://kexue.fm/archives/11480)

[^k11494]: 苏剑林. 让炼丹更科学一些（四）：新恒等式，新学习率. kexue.fm 2025. [https://kexue.fm/archives/11494](https://kexue.fm/archives/11494)

[^k11530]: 苏剑林. 让炼丹更科学一些（五）：基于梯度精调学习率. kexue.fm 2026. [https://kexue.fm/archives/11530](https://kexue.fm/archives/11530)

[^k11540]: 苏剑林. 让炼丹更科学一些（六）：自上而下的精妙构造. kexue.fm 2026. [https://kexue.fm/archives/11540](https://kexue.fm/archives/11540)

[^k11804]: 苏剑林. 让炼丹更科学一些（七）：步长调度与权重平均. kexue.fm 2026. [https://kexue.fm/archives/11804](https://kexue.fm/archives/11804)

[^k11879]: 苏剑林. 让炼丹更科学一些（八）：多阶段训练的学习率. kexue.fm 2026. [https://kexue.fm/archives/11879](https://kexue.fm/archives/11879)

[^k11882]: 苏剑林. 让炼丹更科学一些（九）：经典自适应梯度算法. kexue.fm 2026. [https://kexue.fm/archives/11882](https://kexue.fm/archives/11882)

[^k7787]: 苏剑林. 从动力学角度看优化算法（五）：为什么学习率不宜过小？ kexue.fm 2020. [https://kexue.fm/archives/7787](https://kexue.fm/archives/7787)

[^k5655]: 苏剑林. 从动力学角度看优化算法（一）：从SGD到动量加速. kexue.fm 2018. [https://kexue.fm/archives/5655](https://kexue.fm/archives/5655)

[^k6234]: 苏剑林. 从动力学角度看优化算法（二）：自适应学习率算法. kexue.fm 2019. [https://kexue.fm/archives/6234](https://kexue.fm/archives/6234)

[^k6261]: 苏剑林. 从动力学角度看优化算法（三）：一个更整体的视角. kexue.fm 2019. [https://kexue.fm/archives/6261](https://kexue.fm/archives/6261)

[^k6583]: 苏剑林. 从动力学角度看优化算法（四）：GAN的第三个阶段. kexue.fm 2019. [https://kexue.fm/archives/6583](https://kexue.fm/archives/6583)

[^k7980]: 苏剑林. 从动力学角度看优化算法（六）：为什么SimSiam不退化？ kexue.fm 2021. [https://kexue.fm/archives/7980](https://kexue.fm/archives/7980)

[^k8009]: 苏剑林. 从动力学角度看优化算法（七）：SGD ≈ SVM？ kexue.fm 2021. [https://kexue.fm/archives/8009](https://kexue.fm/archives/8009)

[^k9660]: 苏剑林. 梯度流：探索通向最小值之路. kexue.fm 2023. [https://kexue.fm/archives/9660](https://kexue.fm/archives/9660)

[^k9359]: 苏剑林. 用热传导方程来指导自监督学习. kexue.fm 2022. [https://kexue.fm/archives/9359](https://kexue.fm/archives/9359)

[^k7521]: 苏剑林. 从采样看优化：可导优化与不可导优化的统一视角. kexue.fm 2020. [https://kexue.fm/archives/7521](https://kexue.fm/archives/7521)

[^k7737]: 苏剑林. 殊途同归的策略梯度与零阶优化. kexue.fm 2020. [https://kexue.fm/archives/7737](https://kexue.fm/archives/7737)

[^k4277]: 苏剑林. 梯度下降和EM算法：系出同源，一脉相承. kexue.fm 2017. [https://kexue.fm/archives/4277](https://kexue.fm/archives/4277)

[^k5239]: 苏剑林. 从最大似然到EM算法：一致的理解方式. kexue.fm 2018. [https://kexue.fm/archives/5239](https://kexue.fm/archives/5239)

[^k5617]: 苏剑林. “噪声对比估计”杂谈：曲径通幽之妙. kexue.fm 2018. [https://kexue.fm/archives/5617](https://kexue.fm/archives/5617)

[^k8968]: 苏剑林. 指数梯度下降 + 元学习 = 自适应学习率. kexue.fm 2022. [https://kexue.fm/archives/8968](https://kexue.fm/archives/8968)

[^k10592]: 苏剑林. Muon优化器赏析：从向量到矩阵的本质跨越. kexue.fm 2024. [https://kexue.fm/archives/10592](https://kexue.fm/archives/10592)

[^k10648]: 苏剑林. 从谱范数梯度到新式权重衰减的思考. kexue.fm 2024. [https://kexue.fm/archives/10648](https://kexue.fm/archives/10648)

[^k10739]: 苏剑林. Muon续集：为什么我们选择尝试Muon？ kexue.fm 2025. [https://kexue.fm/archives/10739](https://kexue.fm/archives/10739)

[^k11416]: 苏剑林. Muon优化器指南：快速上手与关键细节. kexue.fm 2025. [https://kexue.fm/archives/11416](https://kexue.fm/archives/11416)

[^k11549]: 苏剑林. 为什么我们偏爱各向同性？基于最速下降的理解. kexue.fm 2026. [https://kexue.fm/archives/11549](https://kexue.fm/archives/11549)

[^k11875]: 苏剑林. 动量的新理解：逼近特征层面的梯度下降. kexue.fm 2026. [https://kexue.fm/archives/11875](https://kexue.fm/archives/11875)

[^k11126]: 苏剑林. QK-Clip：让Muon在Scaleup之路上更进一步. kexue.fm 2025. [https://kexue.fm/archives/11126](https://kexue.fm/archives/11126)

[^k10795]: 苏剑林. 高阶MuP：更简明但更高明的谱条件缩放. kexue.fm 2025. [https://kexue.fm/archives/10795](https://kexue.fm/archives/10795)

[^k11772]: 苏剑林. 为什么官方版Muon比MuP版多出一个max(1, ⋅)？ kexue.fm 2026. [https://kexue.fm/archives/11772](https://kexue.fm/archives/11772)

[^k10847]: 苏剑林. 矩阵的有效秩（Effective Rank）. kexue.fm 2025. [https://kexue.fm/archives/10847](https://kexue.fm/archives/10847)

[^k11767]: 苏剑林. 矩阵参数的奇异值熵越高越好吗？ kexue.fm 2026. [https://kexue.fm/archives/11767](https://kexue.fm/archives/11767)

[^k11784]: 苏剑林. 强制间隔投影（Margin-Enforcing Projection）. kexue.fm 2026. [https://kexue.fm/archives/11784](https://kexue.fm/archives/11784)

[^k10922]: 苏剑林. msign算子的Newton-Schulz迭代（上）. kexue.fm 2025. [https://kexue.fm/archives/10922](https://kexue.fm/archives/10922)

[^k10996]: 苏剑林. msign算子的Newton-Schulz迭代（下）. kexue.fm 2025. [https://kexue.fm/archives/10996](https://kexue.fm/archives/10996)

[^k11006]: 苏剑林. 通过msign来计算奇异值裁剪mclip（上）. kexue.fm 2025. [https://kexue.fm/archives/11006](https://kexue.fm/archives/11006)

[^k11025]: 苏剑林. msign的导数. kexue.fm 2025. [https://kexue.fm/archives/11025](https://kexue.fm/archives/11025)

[^k11059]: 苏剑林. 通过msign来计算奇异值裁剪mclip（下）. kexue.fm 2025. [https://kexue.fm/archives/11059](https://kexue.fm/archives/11059)

[^k11335]: 苏剑林. 随机矩阵的谱范数的快速估计. kexue.fm 2025. [https://kexue.fm/archives/11335](https://kexue.fm/archives/11335)

[^k11736]: 苏剑林. 如何更科学地估计矩阵的谱范数？ kexue.fm 2026. [https://kexue.fm/archives/11736](https://kexue.fm/archives/11736)

[^k11654]: 苏剑林. 基于流式幂迭代的Muon实现：1. 初识. kexue.fm 2026. [https://kexue.fm/archives/11654](https://kexue.fm/archives/11654)

[^k11673]: 苏剑林. 基于流式幂迭代的Muon实现：2. 加速. kexue.fm 2026. [https://kexue.fm/archives/11673](https://kexue.fm/archives/11673)

[^k11697]: 苏剑林. 基于流式幂迭代的Muon实现：3. 雕琢. kexue.fm 2026. [https://kexue.fm/archives/11697](https://kexue.fm/archives/11697)

[^k11710]: 苏剑林. 基于流式幂迭代的Muon实现：4. 原理. kexue.fm 2026. [https://kexue.fm/archives/11710](https://kexue.fm/archives/11710)

[^k11719]: 苏剑林. 基于流式幂迭代的Muon实现：5. 延伸. kexue.fm 2026. [https://kexue.fm/archives/11719](https://kexue.fm/archives/11719)

[^k11196]: 苏剑林. 流形上的最速下降：1. SGD + 超球面. kexue.fm 2025. [https://kexue.fm/archives/11196](https://kexue.fm/archives/11196)

[^k11215]: 苏剑林. 流形上的最速下降：2. Muon + 正交. kexue.fm 2025. [https://kexue.fm/archives/11215](https://kexue.fm/archives/11215)

[^k11221]: 苏剑林. 流形上的最速下降：3. Muon + Stiefel. kexue.fm 2025. [https://kexue.fm/archives/11221](https://kexue.fm/archives/11221)

[^k11241]: 苏剑林. 流形上的最速下降：4. Muon + 谱球面. kexue.fm 2025. [https://kexue.fm/archives/11241](https://kexue.fm/archives/11241)

[^k11260]: 苏剑林. 重新思考学习率与Batch Size（一）：现状. kexue.fm 2025. [https://kexue.fm/archives/11260](https://kexue.fm/archives/11260)

[^k11280]: 苏剑林. 重新思考学习率与Batch Size（二）：平均场. kexue.fm 2025. [https://kexue.fm/archives/11280](https://kexue.fm/archives/11280)

[^k11285]: 苏剑林. 重新思考学习率与Batch Size（三）：Muon. kexue.fm 2025. [https://kexue.fm/archives/11285](https://kexue.fm/archives/11285)

[^k11301]: 苏剑林. 重新思考学习率与Batch Size（四）：EMA. kexue.fm 2025. [https://kexue.fm/archives/11301](https://kexue.fm/archives/11301)

[^k10542]: 苏剑林. 当Batch Size增大时，学习率该如何随之变化？ kexue.fm 2024. [https://kexue.fm/archives/10542](https://kexue.fm/archives/10542)

[^k10563]: 苏剑林. Adam的epsilon如何影响学习率的Scaling Law？ kexue.fm 2024. [https://kexue.fm/archives/10563](https://kexue.fm/archives/10563)

[^k11833]: 苏剑林. 解构Scaling Law：优化、架构、数据的三重奏. kexue.fm 2026. [https://kexue.fm/archives/11833](https://kexue.fm/archives/11833)

[^k9607]: 苏剑林. 基于量子化假设推导模型的尺度定律（Scaling Law）. kexue.fm 2023. [https://kexue.fm/archives/9607](https://kexue.fm/archives/9607)

[^k11267]: 苏剑林. 为什么Adam的Update RMS是0.2？ kexue.fm 2025. [https://kexue.fm/archives/11267](https://kexue.fm/archives/11267)

[^k11307]: 苏剑林. AdamW的Weight RMS的渐近估计（上）. kexue.fm 2025. [https://kexue.fm/archives/11307](https://kexue.fm/archives/11307)

[^k11404]: 苏剑林. AdamW的Weight RMS的渐近估计（下）. kexue.fm 2025. [https://kexue.fm/archives/11404](https://kexue.fm/archives/11404)

[^k11459]: 苏剑林. 滑动平均视角下的权重衰减和学习率. kexue.fm 2025. [https://kexue.fm/archives/11459](https://kexue.fm/archives/11459)

[^k11593]: 苏剑林. Adam优化器的最优超参数是β1=β2 ？ kexue.fm 2026. [https://kexue.fm/archives/11593](https://kexue.fm/archives/11593)

[^k10588]: 苏剑林. 从Hessian近似看自适应学习率优化器. kexue.fm 2024. [https://kexue.fm/archives/10588](https://kexue.fm/archives/10588)

[^k7469]: 苏剑林. 为什么梯度裁剪能加速训练过程？一个简明的分析. kexue.fm 2020. [https://kexue.fm/archives/7469](https://kexue.fm/archives/7469)

[^k10657]: 苏剑林. 为什么梯度裁剪的默认模长是1？ kexue.fm 2025. [https://kexue.fm/archives/10657](https://kexue.fm/archives/10657)

[^k7888]: 苏剑林. 也来谈谈RNN的梯度消失/爆炸问题. kexue.fm 2020. [https://kexue.fm/archives/7888](https://kexue.fm/archives/7888)

[^k8796]: 苏剑林. 输入梯度惩罚与参数梯度惩罚的一个不等式. kexue.fm 2021. [https://kexue.fm/archives/8796](https://kexue.fm/archives/8796)

[^k6794]: 苏剑林. 用时间换取效果：Keras梯度累积优化器. kexue.fm 2019. [https://kexue.fm/archives/6794](https://kexue.fm/archives/6794)

[^k8634]: 苏剑林. 隐藏在动量中的梯度累积：少更新几步，效果反而更好？ kexue.fm 2021. [https://kexue.fm/archives/8634](https://kexue.fm/archives/8634)

[^k6869]: 苏剑林. Keras实现两个优化器：Lookahead和LazyOptimizer. kexue.fm 2019. [https://kexue.fm/archives/6869](https://kexue.fm/archives/6869)

[^k7094]: 苏剑林. 6个派生优化器的简单介绍及其实现. kexue.fm 2019. [https://kexue.fm/archives/7094](https://kexue.fm/archives/7094)

[^k7302]: 苏剑林. AdaFactor优化器浅析（附开源实现）. kexue.fm 2020. [https://kexue.fm/archives/7302](https://kexue.fm/archives/7302)

[^k7387]: 苏剑林. AdaX优化器浅析（附开源实现）. kexue.fm 2020. [https://kexue.fm/archives/7387](https://kexue.fm/archives/7387)

[^k9473]: 苏剑林. Google新搜出的优化器Lion：效率与效果兼得的“训练狮”. kexue.fm 2023. [https://kexue.fm/archives/9473](https://kexue.fm/archives/9473)

[^k9512]: 苏剑林. Tiger：一个“抠”到极致的优化器. kexue.fm 2023. [https://kexue.fm/archives/9512](https://kexue.fm/archives/9512)

[^k7196]: 苏剑林. 你的CRF层的学习率可能不够大. kexue.fm 2020. [https://kexue.fm/archives/7196](https://kexue.fm/archives/7196)

[^k9344]: 苏剑林. 基于Amos优化器思想推导出来的一些“炼丹策略”. kexue.fm 2022. [https://kexue.fm/archives/9344](https://kexue.fm/archives/9344)

[^k9590]: 苏剑林. 梯度视角下的LoRA：简介、分析、猜测及推广. kexue.fm 2023. [https://kexue.fm/archives/9590](https://kexue.fm/archives/9590)

[^k8870]: 苏剑林. 多任务学习漫谈（一）：以损失之名. kexue.fm 2022. [https://kexue.fm/archives/8870](https://kexue.fm/archives/8870)

[^k8896]: 苏剑林. 多任务学习漫谈（二）：行梯度之事. kexue.fm 2022. [https://kexue.fm/archives/8896](https://kexue.fm/archives/8896)

[^k8907]: 苏剑林. 多任务学习漫谈（三）：分主次之序. kexue.fm 2022. [https://kexue.fm/archives/8907](https://kexue.fm/archives/8907)

[^k10226]: 苏剑林. 对齐全量微调！这是我看过最精彩的LoRA改进（一）. kexue.fm 2024. [https://kexue.fm/archives/10226](https://kexue.fm/archives/10226)

[^k10266]: 苏剑林. 对齐全量微调！这是我看过最精彩的LoRA改进（二）. kexue.fm 2024. [https://kexue.fm/archives/10266](https://kexue.fm/archives/10266)

[^k10001]: 苏剑林. 配置不同的学习率，LoRA还能再涨一点？ kexue.fm 2024. [https://kexue.fm/archives/10001](https://kexue.fm/archives/10001)

[^k10489]: 苏剑林. VQ的旋转技巧：梯度直通估计的一般推广. kexue.fm 2024. [https://kexue.fm/archives/10489](https://kexue.fm/archives/10489)

[^k10770]: 苏剑林. 初探MuP：超参数的跨模型尺度迁移规律. kexue.fm 2025. [https://kexue.fm/archives/10770](https://kexue.fm/archives/10770)

[^k11340]: 苏剑林. MuP之上：1. 好模型的三个特征. kexue.fm 2025. [https://kexue.fm/archives/11340](https://kexue.fm/archives/11340)

[^k11605]: 苏剑林. MuP之上：2. 线性层与最速下降. kexue.fm 2026. [https://kexue.fm/archives/11605](https://kexue.fm/archives/11605)

[^k11647]: 苏剑林. MuP之上：3. 特殊情况特殊处理. kexue.fm 2026. [https://kexue.fm/archives/11647](https://kexue.fm/archives/11647)

[^k11729]: 苏剑林. MuP之上：4. 坚守参数的稳定性. kexue.fm 2026. [https://kexue.fm/archives/11729](https://kexue.fm/archives/11729)

[^k8620]: 苏剑林. 浅谈Transformer的初始化、参数化与标准化. kexue.fm 2021. [https://kexue.fm/archives/8620](https://kexue.fm/archives/8620)

[^k8747]: 苏剑林. 模型优化漫谈：BERT的初始标准差为什么是0.02？ kexue.fm 2021. [https://kexue.fm/archives/8747](https://kexue.fm/archives/8747)

[^k7180]: 苏剑林. 从几何视角来理解模型参数的初始化策略. kexue.fm 2020. [https://kexue.fm/archives/7180](https://kexue.fm/archives/7180)

[^k6992]: 苏剑林. BN究竟起了什么作用？一个闭门造车的分析. kexue.fm 2019. [https://kexue.fm/archives/6992](https://kexue.fm/archives/6992)

[^k10831]: 苏剑林. 通过梯度近似寻找Normalization的替代品. kexue.fm 2025. [https://kexue.fm/archives/10831](https://kexue.fm/archives/10831)

[^k9119]: 苏剑林. 生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼. kexue.fm 2022. [https://kexue.fm/archives/9119](https://kexue.fm/archives/9119)

[^k9152]: 苏剑林. 生成扩散模型漫谈（二）：DDPM = 自回归式VAE. kexue.fm 2022. [https://kexue.fm/archives/9152](https://kexue.fm/archives/9152)

[^k9164]: 苏剑林. 生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪. kexue.fm 2022. [https://kexue.fm/archives/9164](https://kexue.fm/archives/9164)

[^k9181]: 苏剑林. 生成扩散模型漫谈（四）：DDIM = 高观点DDPM. kexue.fm 2022. [https://kexue.fm/archives/9181](https://kexue.fm/archives/9181)

[^k9209]: 苏剑林. 生成扩散模型漫谈（五）：一般框架之SDE篇. kexue.fm 2022. [https://kexue.fm/archives/9209](https://kexue.fm/archives/9209)

[^k9228]: 苏剑林. 生成扩散模型漫谈（六）：一般框架之ODE篇. kexue.fm 2022. [https://kexue.fm/archives/9228](https://kexue.fm/archives/9228)

[^k9280]: 苏剑林. 生成扩散模型漫谈（十二）：“硬刚”扩散ODE. kexue.fm 2022. [https://kexue.fm/archives/9280](https://kexue.fm/archives/9280)

[^k9305]: 苏剑林. 生成扩散模型漫谈（十三）：从万有引力到扩散模型. kexue.fm 2022. [https://kexue.fm/archives/9305](https://kexue.fm/archives/9305)

[^k9370]: 苏剑林. 生成扩散模型漫谈（十四）：构建ODE的一般步骤（上）. kexue.fm 2025. [https://kexue.fm/archives/9370](https://kexue.fm/archives/9370)

[^k9379]: 苏剑林. 生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）. kexue.fm 2025. [https://kexue.fm/archives/9379](https://kexue.fm/archives/9379)

[^k9497]: 苏剑林. 生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）. kexue.fm 2025. [https://kexue.fm/archives/9497](https://kexue.fm/archives/9497)

[^k9467]: 苏剑林. 生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配. kexue.fm 2023. [https://kexue.fm/archives/9467](https://kexue.fm/archives/9467)

[^k9509]: 苏剑林. 生成扩散模型漫谈（十八）：得分匹配 = 条件得分匹配. kexue.fm 2023. [https://kexue.fm/archives/9509](https://kexue.fm/archives/9509)

[^k9662]: 苏剑林. 生成扩散模型漫谈（十九）：作为扩散ODE的GAN. kexue.fm 2024. [https://kexue.fm/archives/9662](https://kexue.fm/archives/9662)

[^k9668]: 苏剑林. 生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP. kexue.fm 2024. [https://kexue.fm/archives/9668](https://kexue.fm/archives/9668)

[^k9881]: 苏剑林. 生成扩散模型漫谈（二十一）：中值定理加速ODE采样. kexue.fm 2024. [https://kexue.fm/archives/9881](https://kexue.fm/archives/9881)

[^k10047]: 苏剑林. 生成扩散模型漫谈（二十二）：信噪比与大图生成（上）. kexue.fm 2024. [https://kexue.fm/archives/10047](https://kexue.fm/archives/10047)

[^k10055]: 苏剑林. 生成扩散模型漫谈（二十三）：信噪比与大图生成（下）. kexue.fm 2024. [https://kexue.fm/archives/10055](https://kexue.fm/archives/10055)

[^k10077]: 苏剑林. 生成扩散模型漫谈（二十四）：少走捷径，更快到达. kexue.fm 2024. [https://kexue.fm/archives/10077](https://kexue.fm/archives/10077)

[^k10085]: 苏剑林. 生成扩散模型漫谈（二十五）：基于恒等式的蒸馏（上）. kexue.fm 2024. [https://kexue.fm/archives/10085](https://kexue.fm/archives/10085)

[^k10567]: 苏剑林. 生成扩散模型漫谈（二十六）：基于恒等式的蒸馏（下）. kexue.fm 2025. [https://kexue.fm/archives/10567](https://kexue.fm/archives/10567)

[^k10617]: 苏剑林. 生成扩散模型漫谈（二十七）：将步长作为条件输入. kexue.fm 2025. [https://kexue.fm/archives/10617](https://kexue.fm/archives/10617)

[^k10633]: 苏剑林. 生成扩散模型漫谈（二十八）：分步理解一致性模型. kexue.fm 2025. [https://kexue.fm/archives/10633](https://kexue.fm/archives/10633)

[^k10711]: 苏剑林. 生成扩散模型漫谈（二十九）：用DDPM来离散编码. kexue.fm 2025. [https://kexue.fm/archives/10711](https://kexue.fm/archives/10711)

[^k10958]: 苏剑林. 生成扩散模型漫谈（三十）：从瞬时速度到平均速度. kexue.fm 2025. [https://kexue.fm/archives/10958](https://kexue.fm/archives/10958)

[^k11428]: 苏剑林. 生成扩散模型漫谈（三十一）：预测数据而非噪声. kexue.fm 2026. [https://kexue.fm/archives/11428](https://kexue.fm/archives/11428)

[^k4439]: 苏剑林. 互怼的艺术：从零直达WGAN-GP. kexue.fm 2017. [https://kexue.fm/archives/4439](https://kexue.fm/archives/4439)

[^k6016]: 苏剑林. f-GAN简介：GAN模型的生产车间. kexue.fm 2018. [https://kexue.fm/archives/6016](https://kexue.fm/archives/6016)

[^k6051]: 苏剑林. 深度学习中的Lipschitz约束：泛化与生成模型. kexue.fm 2018. [https://kexue.fm/archives/6051](https://kexue.fm/archives/6051)

[^k6280]: 苏剑林. 从Wasserstein距离、对偶理论到WGAN. kexue.fm 2019. [https://kexue.fm/archives/6280](https://kexue.fm/archives/6280)

[^k8244]: 苏剑林. WGAN的成功，可能跟Wasserstein距离没啥关系. kexue.fm 2021. [https://kexue.fm/archives/8244](https://kexue.fm/archives/8244)

[^k5716]: 苏剑林. 用变分推断统一理解生成模型（VAE、GAN、AAE、ALI）. kexue.fm 2018. [https://kexue.fm/archives/5716](https://kexue.fm/archives/5716)

[^k6316]: 苏剑林. 能量视角下的GAN模型（一）：GAN＝“挖坑”＋“跳坑”. kexue.fm 2019. [https://kexue.fm/archives/6316](https://kexue.fm/archives/6316)

[^k6331]: 苏剑林. 能量视角下的GAN模型（二）：GAN＝“分析”＋“采样”. kexue.fm 2019. [https://kexue.fm/archives/6331](https://kexue.fm/archives/6331)

[^k6612]: 苏剑林. 能量视角下的GAN模型（三）：生成模型=能量模型. kexue.fm 2019. [https://kexue.fm/archives/6612](https://kexue.fm/archives/6612)

[^k6705]: 苏剑林. 漫谈重参数：从正态分布到Gumbel Softmax. kexue.fm 2019. [https://kexue.fm/archives/6705](https://kexue.fm/archives/6705)

[^k10289]: 苏剑林. 通向最优分布之路：概率空间的最小化. kexue.fm 2024. [https://kexue.fm/archives/10289](https://kexue.fm/archives/10289)

[^k8512]: 苏剑林. 两个多元正态分布的KL散度、巴氏距离和W距离. kexue.fm 2021. [https://kexue.fm/archives/8512](https://kexue.fm/archives/8512)

[^k9039]: 苏剑林. GlobalPointer下的“KL散度”应该是怎样的？ kexue.fm 2022. [https://kexue.fm/archives/9039](https://kexue.fm/archives/9039)

[^k11854]: 苏剑林. 除了交叉熵，LM Loss还有什么选择？ kexue.fm 2026. [https://kexue.fm/archives/11854](https://kexue.fm/archives/11854)

[^k10145]: 苏剑林. 通向概率分布之路：盘点Softmax及其替代品. kexue.fm 2024. [https://kexue.fm/archives/10145](https://kexue.fm/archives/10145)

[^k11814]: 苏剑林. LogSumExp和Softmax的泰勒展开. kexue.fm 2026. [https://kexue.fm/archives/11814](https://kexue.fm/archives/11814)

[^k9526]: 苏剑林. 缓解交叉熵过度自信的一个简明方案. kexue.fm 2023. [https://kexue.fm/archives/9526](https://kexue.fm/archives/9526)

[^k9098]: 苏剑林. 如何训练你的准确率？ kexue.fm 2022. [https://kexue.fm/archives/9098](https://kexue.fm/archives/9098)

[^k9797]: 苏剑林. EMO：基于最优传输思想设计的分类损失函数. kexue.fm 2023. [https://kexue.fm/archives/9797](https://kexue.fm/archives/9797)

[^k4733]: 苏剑林. 从loss的硬截断、软化到focal loss. kexue.fm 2017. [https://kexue.fm/archives/4733](https://kexue.fm/archives/4733)

[^k8823]: 苏剑林. 从熵不变性看Attention的Scale操作. kexue.fm 2021. [https://kexue.fm/archives/8823](https://kexue.fm/archives/8823)

[^k9034]: 苏剑林. 熵不变性Softmax的一个快速推导. kexue.fm 2022. [https://kexue.fm/archives/9034](https://kexue.fm/archives/9034)

[^k9588]: 苏剑林. 从JL引理看熵不变性Attention. kexue.fm 2023. [https://kexue.fm/archives/9588](https://kexue.fm/archives/9588)

[^k9812]: 苏剑林. 从梯度最大化看Attention的Scale操作. kexue.fm 2023. [https://kexue.fm/archives/9812](https://kexue.fm/archives/9812)

[^k5448]: 苏剑林. 最小熵原理（一）：无监督学习的原理. kexue.fm 2018. [https://kexue.fm/archives/5448](https://kexue.fm/archives/5448)

[^k5476]: 苏剑林. 最小熵原理（二）：“当机立断”之词库构建. kexue.fm 2018. [https://kexue.fm/archives/5476](https://kexue.fm/archives/5476)

[^k5577]: 苏剑林. 最小熵原理（三）：“飞象过河”之句模版和语言结构. kexue.fm 2018. [https://kexue.fm/archives/5577](https://kexue.fm/archives/5577)

[^k6191]: 苏剑林. 最小熵原理（四）：“物以类聚”之从图书馆到词向量. kexue.fm 2019. [https://kexue.fm/archives/6191](https://kexue.fm/archives/6191)

[^k7695]: 苏剑林. 最小熵原理（六）：词向量的维度应该怎么选择？ kexue.fm 2020. [https://kexue.fm/archives/7695](https://kexue.fm/archives/7695)

[^k9595]: 苏剑林. 如何度量数据的稀疏程度？ kexue.fm 2023. [https://kexue.fm/archives/9595](https://kexue.fm/archives/9595)

[^k8679]: 苏剑林. 让人惊叹的Johnson-Lindenstrauss引理：理论篇. kexue.fm 2021. [https://kexue.fm/archives/8679](https://kexue.fm/archives/8679)

[^k8706]: 苏剑林. 让人惊叹的Johnson-Lindenstrauss引理：应用篇. kexue.fm 2021. [https://kexue.fm/archives/8706](https://kexue.fm/archives/8706)

[^k3534]: 苏剑林. “熵”不起：从熵、最大熵原理到最大熵模型（一）. kexue.fm 2015. [https://kexue.fm/archives/3534](https://kexue.fm/archives/3534)

[^k6181]: 苏剑林. 从变分编码、信息瓶颈到正态分布：论遗忘的重要性. kexue.fm 2018. [https://kexue.fm/archives/6181](https://kexue.fm/archives/6181)

[^k8578]: 苏剑林. 概率视角下的线性模型：逻辑回归有解析解吗？ kexue.fm 2021. [https://kexue.fm/archives/8578](https://kexue.fm/archives/8578)

[^k6747]: 苏剑林. 简述无偏估计和有偏估计. kexue.fm 2019. [https://kexue.fm/archives/6747](https://kexue.fm/archives/6747)

[^k8978]: 苏剑林. 训练1000层的Transformer究竟有什么困难？ kexue.fm 2022. [https://kexue.fm/archives/8978](https://kexue.fm/archives/8978)

[^k8994]: 苏剑林. 为什么需要残差？一个来自DeepNet的视角. kexue.fm 2022. [https://kexue.fm/archives/8994](https://kexue.fm/archives/8994)

[^k9009]: 苏剑林. 为什么Pre Norm的效果不如Post Norm？ kexue.fm 2022. [https://kexue.fm/archives/9009](https://kexue.fm/archives/9009)

[^k8444]: 苏剑林. 我们可以无损放大一个Transformer模型吗（一）. kexue.fm 2021. [https://kexue.fm/archives/8444](https://kexue.fm/archives/8444)

[^k11371]: 苏剑林. 低精度Attention可能存在有偏的舍入误差. kexue.fm 2025. [https://kexue.fm/archives/11371](https://kexue.fm/archives/11371)

[^k8128]: 苏剑林. Seq2Seq重复解码现象的理论分析尝试. kexue.fm 2021. [https://kexue.fm/archives/8128](https://kexue.fm/archives/8128)

[^k7259]: 苏剑林. Seq2Seq中Exposure Bias现象的浅析与对策. kexue.fm 2020. [https://kexue.fm/archives/7259](https://kexue.fm/archives/7259)

[^k7500]: 苏剑林. 如何应对Seq2Seq中的“根本停不下来”问题？ kexue.fm 2020. [https://kexue.fm/archives/7500](https://kexue.fm/archives/7500)

[^k9617]: 苏剑林. NBCE：使用朴素贝叶斯扩展LLM的Context处理长度. kexue.fm 2023. [https://kexue.fm/archives/9617](https://kexue.fm/archives/9617)

[^k7430]: 苏剑林. Google新作Synthesizer：我们还不够了解自注意力. kexue.fm 2020. [https://kexue.fm/archives/7430](https://kexue.fm/archives/7430)

[^k9019]: 苏剑林. 听说Attention与Softmax更配哦～ kexue.fm 2022. [https://kexue.fm/archives/9019](https://kexue.fm/archives/9019)

[^k9889]: 苏剑林. 注意力机制真的可以“集中注意力”吗？ kexue.fm 2023. [https://kexue.fm/archives/9889](https://kexue.fm/archives/9889)

[^k11664]: 苏剑林. Attention Residuals 回忆录. kexue.fm 2026. [https://kexue.fm/archives/11664](https://kexue.fm/archives/11664)

[^k10017]: 苏剑林. 时空之章：将Attention视为平方复杂度的RNN. kexue.fm 2024. [https://kexue.fm/archives/10017](https://kexue.fm/archives/10017)

[^k10373]: 苏剑林. Softmax后传：寻找Top-K的光滑近似. kexue.fm 2024. [https://kexue.fm/archives/10373](https://kexue.fm/archives/10373)

[^k8069]: 苏剑林. 你可能不需要BERT-flow：一个线性变换媲美BERT-flow. kexue.fm 2021. [https://kexue.fm/archives/8069](https://kexue.fm/archives/8069)

[^k9079]: 苏剑林. 当BERT-whitening引入超参数：总有一款适合你. kexue.fm 2022. [https://kexue.fm/archives/9079](https://kexue.fm/archives/9079)

[^k9698]: 苏剑林. 语言模型输出端共享Embedding的重新探索. kexue.fm 2023. [https://kexue.fm/archives/9698](https://kexue.fm/archives/9698)

[^k9762]: 苏剑林. 大词表语言模型在续写任务上的一个问题及对策. kexue.fm 2023. [https://kexue.fm/archives/9762](https://kexue.fm/archives/9762)

[^k10699]: 苏剑林. MoE环游记：1、从几何意义出发. kexue.fm 2025. [https://kexue.fm/archives/10699](https://kexue.fm/archives/10699)

[^k10815]: 苏剑林. MoE环游记：4、难处应当多投入. kexue.fm 2025. [https://kexue.fm/archives/10815](https://kexue.fm/archives/10815)

[^k11619]: 苏剑林. MoE环游记：6、最优分配促均衡. kexue.fm 2026. [https://kexue.fm/archives/11619](https://kexue.fm/archives/11619)

[^k11760]: 苏剑林. MoE环游记：8、强制序列级均衡. kexue.fm 2026. [https://kexue.fm/archives/11760](https://kexue.fm/archives/11760)

[^k11782]: 苏剑林. MoE环游记：9、门控归一化之争. kexue.fm 2026. [https://kexue.fm/archives/11782](https://kexue.fm/archives/11782)

[^k11320]: 苏剑林. 为什么线性注意力要加Short Conv？ kexue.fm 2025. [https://kexue.fm/archives/11320](https://kexue.fm/archives/11320)

[^k11486]: 苏剑林. 为什么DeltaNet要加L2 Normalize？ kexue.fm 2025. [https://kexue.fm/archives/11486](https://kexue.fm/archives/11486)

[^k11823]: 苏剑林. 将Softmax Attention线性化为Gated DeltaNet. kexue.fm 2026. [https://kexue.fm/archives/11823](https://kexue.fm/archives/11823)

[^k10114]: 苏剑林. 重温SSM（一）：线性系统和HiPPO矩阵. kexue.fm 2024. [https://kexue.fm/archives/10114](https://kexue.fm/archives/10114)

[^k10162]: 苏剑林. 重温SSM（三）：HiPPO的高效计算（S4）. kexue.fm 2024. [https://kexue.fm/archives/10162](https://kexue.fm/archives/10162)

[^k9783]: 苏剑林. 脑洞大开：非线性RNN居然也可以并行计算？ kexue.fm 2023. [https://kexue.fm/archives/9783](https://kexue.fm/archives/9783)

[^k8601]: 苏剑林. Transformer升级之路：5、作为无限维的线性Attention. kexue.fm 2021. [https://kexue.fm/archives/8601](https://kexue.fm/archives/8601)

[^k10091]: 苏剑林. 缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA. kexue.fm 2024. [https://kexue.fm/archives/10091](https://kexue.fm/archives/10091)

[^k10907]: 苏剑林. Transformer升级之路：20、MLA好在哪里?（上）. kexue.fm 2025. [https://kexue.fm/archives/10907](https://kexue.fm/archives/10907)

[^k8373]: 苏剑林. GlobalPointer：用统一的方式处理嵌套和非嵌套NER. kexue.fm 2021. [https://kexue.fm/archives/8373](https://kexue.fm/archives/8373)

[^k8877]: 苏剑林. Efficient GlobalPointer：少点参数，多点效果. kexue.fm 2022. [https://kexue.fm/archives/8877](https://kexue.fm/archives/8877)

[^k4695]: 苏剑林. 果壳中的条件随机场(CRF In A Nutshell). kexue.fm 2017. [https://kexue.fm/archives/4695](https://kexue.fm/archives/4695)

[^k7213]: 苏剑林. CRF用过了，不妨再了解下更快的MEMM？ kexue.fm 2020. [https://kexue.fm/archives/7213](https://kexue.fm/archives/7213)

[^k8671]: 苏剑林. 曾被嫌弃的预训练任务NSP，做出了优秀的Zero Shot效果. kexue.fm 2021. [https://kexue.fm/archives/8671](https://kexue.fm/archives/8671)

[^k9046]: 苏剑林. 你的语言模型有没有“无法预测的词”？ kexue.fm 2022. [https://kexue.fm/archives/9046](https://kexue.fm/archives/9046)

[^k4299]: 苏剑林. 【不可思议的Word2Vec】 1.数学原理. kexue.fm 2017. [https://kexue.fm/archives/4299](https://kexue.fm/archives/4299)

[^k4667]: 苏剑林. 更别致的词向量模型(一)：simpler glove. kexue.fm 2017. [https://kexue.fm/archives/4667](https://kexue.fm/archives/4667)

[^k4675]: 苏剑林. 更别致的词向量模型(四)：模型的求解. kexue.fm 2017. [https://kexue.fm/archives/4675](https://kexue.fm/archives/4675)

[^k8656]: 苏剑林. 从三角不等式到Margin Softmax. kexue.fm 2021. [https://kexue.fm/archives/8656](https://kexue.fm/archives/8656)

[^k9368]: 苏剑林. 从局部到全局：语义相似度的测地线距离. kexue.fm 2022. [https://kexue.fm/archives/9368](https://kexue.fm/archives/9368)

[^k11818]: 苏剑林. 基于排序不等式的相似度指标. kexue.fm 2026. [https://kexue.fm/archives/11818](https://kexue.fm/archives/11818)

[^k7388]: 苏剑林. 从EMD、WMD到WRD：文本向量序列的相似度计算. kexue.fm 2020. [https://kexue.fm/archives/7388](https://kexue.fm/archives/7388)

[^k9336]: 苏剑林. 利用CUR分解加速交互式相似度模型的检索. kexue.fm 2022. [https://kexue.fm/archives/9336](https://kexue.fm/archives/9336)
