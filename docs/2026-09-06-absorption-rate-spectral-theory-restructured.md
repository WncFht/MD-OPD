# OPD 吸收率下降：一个谱动力学解释

本文是对原阅读笔记的重构版。目标不是罗列所有相关文献或实验设想，而是回答一个问题：**为什么 OPD 的吸收率会随训练下降，而且训练节奏对数据量不敏感？**

本文的结论分为三层：

1. 论文已经直接支持：冻结状态后，OPD 仍会持续对齐数百步；不同数据规模的吸收节奏相近；学习率主要重标定时间轴[^fu26]。
2. 在局部二次近似和一阶优化器下，这些现象由一个最小谱模型自然推出：不同曲率方向以不同速率衰减，高曲率方向先被吸收，剩余误差逐渐由低曲率方向主导。
3. 这个模型不是完整机制。Adam 的时变预条件、on-policy 状态漂移、曲率谱漂移，以及距离定义本身，都可能改变观测到的曲线形状。

## 1. 问题与观测

### 1.1 OPD 目标

冻结一批学生生成的状态后，OPD 可看作固定经验目标

$$
J(\theta)=\mathbb{E}_{s\in\mathcal S}
\left[\operatorname{KL}(\pi_\theta(\cdot\mid s)\|\pi_T(\cdot\mid s))\right].
$$

实际训练使用逐 token advantage

$$
A_i=\log\pi_T(y_i\mid s_i)-\log\pi_\theta(y_i\mid s_i),
$$

并可使用 top-$k$ 截断版本[^fu26]。论文的距离是

$$
d_t=\frac1{|\mathcal T_t|}\sum_{i\in\mathcal T_t}|A_i|,
\qquad
v_t=\frac{d_t-d_{t+1}}{d_t}.
$$

其中 $v_t$ 表示一步相对吸收率，而不是绝对损失下降量。

### 1.2 论文给出的约束

论文报告了三个与解释直接相关的事实[^fu26]：

- 1 个 query、4 个 query、16 个 query 和 full-data 的 run，在前 300 步消耗的相对距离相近，$d_{300}/d_{30}$ 约为 $0.16$--$0.22$。
- 学习率为 $0.5,1,2\times10^{-6}$ 时，按 $u=\eta t$ 重标定后曲线大致坍缩，说明学习率主要改变前进速度。
- 固定最初采样的状态集后，训练仍可稳定增长约 200 步。因此，慢节奏不能主要归因于“必须不断获得新状态”。

这些结果把解释对象限定为：**固定目标上的参数空间动力学，以及它受到噪声和优化器的修正。**

### 1.3 为什么要测 $d_t$：它是“还剩多少没有吸收”的可观测代理

$d_t$ 不是训练 loss 的任意替代量，而是对 OPD 核心任务的直接测量：在同一批学生访问状态和 token 上，学生与教师的 log-probability 仍相差多少。对单个 token，

$$
A_i(t)=\log\pi_T(y_i\mid s_i)-\log\pi_{\theta_t}(y_i\mid s_i).
$$

因此 $|A_i|$ 测量的是该 token 上的绝对 log 概率缺口；平均得到的 $d_t$ 是一批固定比较对象上的剩余不一致程度。测量它有四个作用：定义无量纲的相对吸收率 $v_t$，在冻结状态时隔离参数更新动力学，把曲线连接到谱模式的混合，并区分“当前吸收速度”和“最终能达到的误差底”。

但 $d_t$ 不是完整的 reverse-KL：它只在学生实际采样的 token 上取绝对 advantage，并且可能使用 top-$k$ 截断。因此它是面向 OPD 训练轨迹的诊断量，而不是全词表分布距离；比较不同实验时必须固定状态、token 采样和距离口径。

#### 1.3.1 为什么不直接测 top-$k$ KL？为什么要加绝对值？

完整的 token-level reverse-KL 是

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_T\mid s)
=\sum_{v\in\mathcal V}\pi_\theta(v\mid s)
\left[\log\pi_\theta(v\mid s)-\log\pi_T(v\mid s)\right].
$$

它当然比单个 $|A_i|$ 更接近真正的分布距离。OPD 论文实际的 top-$k$ advantage 也不是简单“截断后丢掉尾部”：它先取学生 top-$k$ 集合 $K_s$，再将学生概率在 $K_s$ 上重新归一化，形成

$$
\widehat D_k=\sum_{v\in K_s}\tilde\pi_\theta(v\mid s)\log\frac{\tilde\pi_\theta(v\mid s)}{\pi_T(v\mid s)},
$$

其中 $\tilde\pi_\theta$ 表示在 $K_s$ 上归一化后的学生分布。它的优点是比单 token 估计方差低，代价是只描述学生概率质量的 top-$k$ 部分，不再是完整词表 KL；而且 $K_s$ 会随当前学生改变。故 full/top-$k$ KL 适合衡量分布本身或做低频率审计，$d_t$ 适合在固定 rollout/token 集上逐步追踪“同一批缺口还剩多少”。这不是说 top-$k$ KL 不准确，而是它回答的是不同问题。

加绝对值是因为 $A_i$ 带符号：正值表示教师给该 token 更高概率，负值表示学生更高。直接平均 $\mathbb E[A_i]$ 会让这两类错误相互抵消，甚至在学生仍严重偏离教师时读成接近零；$\mathbb E|A_i|$ 测量的是偏离幅度。它仍不是 KL，因为 KL 对差异按学生概率加权且不取绝对值。因此 $d_t$ 的定位是“固定比较集上的绝对 log-prob 缺口”，不是严格分布距离。

## 2. 最小谱模型

### 2.1 局部几何：reverse-KL 的 Hessian、Fisher 与可表示性

冻结状态集后，令 $p_s(y)=\pi_T(y\mid s)$、$q_\theta(y)=\pi_\theta(y\mid s)$，并把状态平均吸收到记号中：

$$
J(\theta)=\mathbb E_s D_{\rm KL}(q_\theta\Vert p_s).
$$

先讨论可表示情形：存在 $\theta^*$ 使 $q_{\theta^*}=p_s$（在所有冻结状态上）。记 $u_\theta(y)=\nabla_\theta\log q_\theta(y)$。在 $\theta^*$ 处，score 恒等式给出

$$
\mathbb E_{q_*}[u_*(y)]=0,\qquad
\mathbb E_{q_*}[\nabla_\theta^2\log q_\theta(y)]_{\theta^*}=-F_s,
$$

其中 $F_s=\mathbb E_{q_*}[u_*u_*^\top]$。令 $\delta=\theta-\theta^*$，对 $\log q_{\theta^*+\delta}-\log q_{\theta^*}$ 做 Taylor 展开，并注意 KL 中的积分测度也是 $q_{\theta^*+\delta}$，可得

$$
D_{\rm KL}(q_{\theta^*+\delta}\Vert q_{\theta^*})
=\frac12\delta^\top F_s\delta+O(\|\delta\|^3).
$$

状态平均后 $F=\mathbb E_sF_s$。因此梯度和 Hessian 分别为 $\nabla J(\theta^*+\delta)=F\delta+O(\|\delta\|^2)$、$H_J(\theta^*)=F$。对 softmax logits（指数族自然参数），这个 Fisher 也等于 generalized Gauss--Newton；该等式是交叉熵/对数似然下的精确结构，而不是把任意 Hessian 都替换成 Fisher 的经验近似[^amari98][^amari00][^martens14]。

若教师不在学生模型族内，取 $\theta^*=\arg\min J$，则 $H_J(\theta^*)$ 一般写成

$$
H_J=F+R,\qquad
R=\mathbb E_s\mathbb E_{q_{\theta^*}}
\left[\big(\log q_{\theta^*}-\log p_s\big)
\left(\nabla_\theta^2\log q_\theta+u_\theta u_\theta^\top\right)\right]_{\theta^*}.
$$

残差项 $R$ 只有在匹配或局部小失配时才可忽略；这正是“表示缺口”会在长尾留下平台的几何来源。实际测量应区分真 Fisher、empirical Fisher 与 Hessian：前者由模型分布的 score 协方差定义，后两者在有限样本和非零失配时并不相同。

对称半正定的局部矩阵 $H$ 做谱分解 $He_i=\lambda_i e_i$、$\delta_t=\sum_i x_i(t)e_i$。普通 GD 的线性化更新为

$$
x_i(t+1)=(1-\eta\lambda_i)x_i(t),\qquad
x_i(t)=q_i^t x_i(0),\quad q_i=1-\eta\lambda_i.
$$

稳定性需要 $|q_i|<1$（即 $0<\eta\lambda_i<2$）；在 OPD 的小步长区间 $\eta\lambda_i\ll1$ 时，$q_i^t\simeq e^{-\eta\lambda_i t}$。若更新为预条件梯度 $\delta_{t+1}=\delta_t-\eta P H\delta_t$，则应将 $\lambda_i$ 理解为 $PH$ 的广义特征值；Adam 只有在 $P_t$ 近似常数时才能退化为此固定谱模型。NTK/lazy-training 理论与此同构：函数误差沿 NTK 特征函数按对应特征值衰减[^jacot18][^arora19][^chizat19][^hu21][^belfer24]。

#### 2.1.1 从二次目标到单方向递推

这一步只用到三件事：局部二次化、梯度下降定义和 Hessian 特征分解。令 $\delta_t=\theta_t-\theta^*$，则

$$
J(\theta^*+\delta_t)=J(\theta^*)+\tfrac12\delta_t^\top H\delta_t,
\qquad \nabla J=H\delta_t.
$$

普通 GD 给出 $\delta_{t+1}=(I-\eta H)\delta_t$。代入 $\delta_t=\sum_i x_i(t)e_i$ 与 $He_i=\lambda_i e_i$：

$$
\delta_{t+1}=\sum_i(1-\eta\lambda_i)x_i(t)e_i,
$$

逐个比较正交基 $e_i$ 的系数，就得到 $x_i(t+1)=(1-\eta\lambda_i)x_i(t)$。因此该递推是代数结果，不是额外的物理假设。Adam 则更接近 $\delta_{t+1}=\delta_t-\eta P_tH_t\delta_t$；只有 $P_t,H_t$ 近似固定时，才存在类似的有效谱方向。

### 2.2 吸收率是剩余误差的平均曲率：离散精确式与连续谱动力学

在线性化区间，把某一距离口径下的剩余误差写成非负模式贡献

$$
d_t=\sum_i a_i q_i^t,\qquad a_i\ge0,\quad q_i=1-\eta\lambda_i\in(0,1).
$$

这里的可加性对 KL 二次型是精确的（$a_i=\tfrac12\lambda_i x_i(0)^2$），对 $\mathbb E|A|$ 则是“符号不发生大量抵消”的一阶近似。论文定义的逐步吸收率因此满足一个**离散精确恒等式**：

$$
v_t=1-\frac{d_{t+1}}{d_t}
=\sum_i w_i(t)(1-q_i)
=\eta\sum_iw_i(t)\lambda_i,
$$

$$
w_i(t)=\frac{a_iq_i^t}{\sum_j a_jq_j^t},\qquad \sum_iw_i(t)=1.
$$

所以在严格二次模型内，$v_t=\eta\langle\lambda\rangle_t$ 并不需要小步长近似；指数形式只是在 $q_i^t\simeq e^{-\eta\lambda_it}$ 后得到的连续时间表示。一步更新的稳定性、梯度裁剪或 momentum 破坏上述等式的条件，应在解释实验曲线时单独检查。

权重满足离散指数倾斜 $w_i(t+1)\propto w_i(t)q_i$。在连续极限 $q_i^t\to e^{-\eta\lambda_it}$ 下，

$$
d(t)=d_0\,\mathbb E_{w_0}[e^{-\eta t\lambda}],\qquad
w(\lambda,t)=\frac{w_0(\lambda)e^{-\eta t\lambda}}
{\mathbb E_{w_0}[e^{-\eta t\lambda}]}.
$$

对剩余谱均值求导：

$$
\frac{d\langle\lambda\rangle_t}{dt}
=-\eta\left(\langle\lambda^2\rangle_t-\langle\lambda\rangle_t^2\right)
=-\eta\,\operatorname{Var}_{w(t)}(\lambda)\le0,
$$

从而

$$
\boxed{\frac{dv^{\rm inst}}{dt}=-\eta^2\operatorname{Var}_{w(t)}(\lambda)\le0},
\qquad v^{\rm inst}=\eta\langle\lambda\rangle_t.
$$

结论的逻辑边界很明确：只要 $a_i\ge0$、$0<\eta\lambda_i<1$ 且谱在观测窗口近似固定，吸收率必然不增；下降斜率的绝对值直接给出剩余谱方差。$\operatorname{Var}=0$ 时只有单一时间尺度，$v_t$ 为常数；谱越宽，早期下降越快。若出现局部上升，至少有一个条件失效：距离的符号抵消、时变预条件/曲率、状态分布漂移、噪声，或步长已接近稳定性边界。

更一般地，预条件更新 $\delta_{t+1}=(I-\eta P H)\delta_t$ 只需把 $\lambda_i$ 换成 $PH$ 的谱，但当 $P=P_t$ 随 Adam 二阶矩变化时，无法再写成固定的 Laplace 变换。这给出一个直接实验判据：若按 $u=\eta t$ 折叠后曲线坍缩，说明有效 $P_tH_t$ 在窗口内变化较小；若坍缩失败且与二阶矩时间常数同步，则应优先归因于优化器而非谱宽。

#### 2.2.1 为什么 $a_i\ge0$？对 $\mathbb E|A|$ 这是不是论文已经证明的？

对二次 KL，非负分解是严格的：

$$
J_t-J^*=\frac12\delta_t^\top H\delta_t
=\frac12\sum_i\lambda_i x_i(t)^2,
$$

所以 $a_i=\tfrac12\lambda_i x_i(0)^2\ge0$。平方消除了模式符号，特征向量正交性消除了交叉项。论文测的是 $d_t=\mathbb E|A_t|$，若 $A_t=\sum_iA_{i,t}$，一般有 $|\sum_iA_{i,t}|\ne\sum_i|A_{i,t}|$；不同模式可能发生正负抵消。因此把 $\mathbb E|A_t|$ 写成 $\sum_i a_iq_i^t$ 是“符号抵消较弱”的有效近似，不是论文已经证明的定理。它的用途是把 L1 曲线连接到谱动力学；验证时应同时记录 $\mathbb E|A|$、$\sqrt{\mathbb E[A^2]}$ 和 $|\mathbb E A|$。

### 2.3 OPD 带来的特殊性质

谱递推本身适用于很多局部二次优化问题；OPD 的特殊性在于它同时改变目标、状态分布和可观测距离：

1. 目标是学生采样状态上的 reverse-KL；不冻结状态时，目标与 $F_t/H_t$ 随学生策略漂移。
2. **OPD 梯度不是普通 CE 的梯度。** 对序列 $y=(y_1,\ldots,y_T)$，Cai et al. 2026[^cai26] 的完整梯度可写成

    $$
    \nabla J_{\rm OPD}
    =\mathbb E_{y\sim\pi_\theta}
    \left[\sum_{t=1}^T\left(\sum_{t'=t}^T\Delta_{t'}\right)
    \nabla\log\pi_\theta(y_t\mid s_t)\right],
    \qquad
    \Delta_{t'}=\log\pi_\theta(y_{t'}\mid s_{t'})-\log\pi_T(y_{t'}\mid s_{t'}).
    $$

    这是因为 $y_{t+1:T}$ 的 log-prob 也依赖于前缀 $y_{1:t}$ 的参数梯度。实践中常用的 immediate-token 近似只保留 $\Delta_t\nabla\log\pi_\theta(y_t\mid s_t)$，从而每个 token 都有 dense 信号。这个“序列累计 advantage → token-level 近似”的结构是 OPD 特有的目标梯度，不是从普通监督标签直接得到的 CE 梯度。

3. 梯度由带符号的 token advantage 估计，top-$k$ 截断会改变有效谱权重，因此 $\mathbb E|A|$ 的非负模式分解只是近似。
4. 教师固定、学生移动，使“剩余误差”有明确的吸收方向；教师与学生共同更新时则不再成立。
5. 学生只在自己访问的状态上训练，状态覆盖缺口和模型表示缺口都可能造成长尾或平台。
6. 学生更新改变后续状态，后续状态又改变 Fisher/Hessian 的加权方式；这是 OPD 特有的 on-policy 反馈。
7. 数据规模若没有明显改变有效 $P_tH_t$ 的谱形状，就可能只改变误差总量而不改变归一化吸收节奏。

与 RL 的对照也可以写成梯度层面的差异。RLVR 通常是

$$
\nabla J_{\rm RL}
=\mathbb E_{y\sim\pi_\theta}\left[(R-b)\sum_{t=1}^T
\nabla\log\pi_\theta(y_t\mid s_t)\right],
$$

其中 $R$ 往往在序列结束时才可得，导致同一个稀疏标量乘遍整条轨迹；OPD 用教师 log-prob 差异提供位置相关的 dense $\Delta_t$（或其后缀和），并且教师项不随学生参数反向传播。因而 OPD 的局部 Hessian 主要来自 token 分布的 Fisher/Gauss--Newton 结构，而 RL 还包含 reward 权重、baseline 方差和策略分布改变带来的曲率项；两者不能只用“都是 on-policy”来等同。

因此，OPD 的更现实局部方程是

$$
\delta_{t+1}\approx\delta_t-\eta P_tH_t\delta_t+\eta\xi_t,
$$

其中 $P_t$ 是 Adam 预条件，$H_t$ 同时包含固定状态曲率、状态漂移和教师不可表示的失配项。固定状态实验压低状态漂移；L1/KL 对照检验距离口径；记录 Adam 二阶矩及 Fisher/Hessian 投影则可判断固定谱近似是否成立。

### 2.4 尾部形状

连续谱写成

$$
d(t)=d_0\int e^{-\eta\lambda t}\,\mu(d\lambda).
$$

它是初始加权谱测度的 Laplace 变换。定义局部幂律指数

$$
\alpha(t)=-\frac{d\log d}{d\log t},
$$

则连续时间下有恒等式

$$
\boxed{\alpha(t)=t\,v^{\rm inst}(t)}.
$$

因此，$v_{50}/v_{200}$、$\alpha(50)$ 和 $\alpha(200)$ 不是独立读数，而应近似满足

$$
\frac{v_{50}}{v_{200}}
\simeq4\frac{\alpha(50)}{\alpha(200)}.
$$

谱的低曲率端决定长尾形状：

- 低曲率处为正则幂律密度时，$d(t)$ 近似幂律，$\alpha(t)$ 近似常数；
- 存在有限带边或谱密度弯曲时，$\alpha(t)$ 会随时间漂移；
- 在 $\lambda=0$ 有质量时，$d(t)$ 逼近平台，$\alpha(t)$ 下降到零。

所以“吸收率下降”是普遍的一阶谱效应；幂律、拉伸指数和平台等定量形状，才是谱结构的指纹。核回归梯度流和早停理论提供了这一 Laplace 谱观点的直接先例[^bordelon20][^yao07][^dieuleveut16]。

## 3. 噪声、优化器与距离口径

### 3.1 随机梯度噪声

单个谱方向可近似为

$$
x_{t+1}=(1-\eta\lambda_i)x_t+\eta\xi_t,
\qquad \mathbb E\xi_t^2=\sigma_i^2/B.
$$

其稳态方差近似为

$$
\mathbb E[x_i^2]_\infty\simeq
\frac{\eta\sigma_i^2}{2\lambda_iB}.
$$

平坦方向收缩慢，同时噪声位移更大，因此存在可吸收性阈值：当初始信号低于该方向的噪声底时，继续增加数据并不会使它表现为稳定的对齐进展。对二次 KL，逐模式的稳态贡献约为

$$
d_{\infty,i}^{\rm KL}\simeq\frac{\eta\sigma_i^2}{4B},
$$

所以总底的可检验标度是 $d_\infty^{\rm KL}\propto\eta/B$，外加与优化无关的表示缺口[^mandt17]。

### 3.2 Adam 不是固定谱上的普通梯度下降

Adam 的坐标更新受梯度符号和二阶矩估计共同影响[^kingma15][^balles18]。当二阶矩仍在累积，或其记忆长度与 run 长度相当时，有效步长会随时间变化；这可以使观测到的 $v_t$ 偏离固定谱模型，甚至产生局部凸起。训练过程中 Hessian 本身也可能发生 progressive sharpening，使 $\lambda_i$ 变成 $\lambda_i(t)$[^cohen21]。

因此，$v_t$ 的局部上升不能直接解释为“谱吸收定律被推翻”。它首先说明至少一个近似失效：固定谱、固定预条件或固定状态分布。论文的学习率折叠支持主时间尺度近似为 $\eta t$，但不保证所有瞬态都按该变量缩放。

### 3.3 L1、KL 与 CE 不是同一个距离

论文读的是 $\mathbb E|A|$，而日志中常用的代理更接近 $|\mathbb EA|$ 或 reverse-KL。二次近似下，KL 距离可写成

$$
d^{\rm KL}(t)\simeq\frac12\sum_i\lambda_i c_i^2e^{-2\eta\lambda_it},
$$

而 L1 距离更接近

$$
d^{\rm L1}(t)\simeq\sum_i a_i e^{-\eta\lambda_it}.
$$

两者有三个结构差异：KL 的模式速率约加倍；KL 额外乘以 $\lambda_i$；$\mathbb E|A|$ 与 $|\mathbb EA|$ 还会受到符号混合影响。此外，reverse-KL 可分解为交叉熵减学生熵，熵变化会显著改变其曲线形状。

因此，跨论文比较 $\alpha$ 或 $v_{50}/v_{200}$ 时，必须先统一距离口径。谱模型的“吸收率递减”结论在同一口径内成立，但不同口径的数值不能直接对比。

## 4. 现有数据的统一解读

我们在 2026-09-06 的两个 run（top-16 与 sampled-token）上得到近似一致的结果：

| 指标             | top-16 | sampled-token |
| ---------------- | -----: | ------------: |
| $d_{279}/d_{30}$ |  0.053 |         0.061 |
| $v_{50}$         |  1.79% |         1.59% |
| $v_{199}$        |  0.59% |         0.63% |
| $v_{50}/v_{200}$ |   3.05 |          2.54 |

这说明在两种 token 估计器下，前进节奏基本相同，支持“节奏主要由 OPD 的参数几何和优化过程决定，而不是 token 子集本身”的判断。

但这些 run 使用的是 KL 代理。其 $\alpha(t)$ 从约 $0.7$ 上升到约 $1.3$，而论文 L1 曲线的局部 $\alpha$ 约为 $0.44$--$0.67$ 且较稳定。这个差异目前只能稳妥地归因于距离口径、熵项、窗口估计器和优化器瞬态的组合，不能直接宣称两组 run 具有不同的谱。

NNLS Laplace 反演也呈现同样的口径依赖：我们的 KL 曲线需要带弯曲的加权谱才能较好外推，而论文 L1 曲线用纯幂律已足够。这个结果支持“训练曲线携带谱信息”，但不单独证明某一种真实 Hessian 谱形状。

## 5. 结论与边界

在固定状态、局部二次化和一阶更新这三个条件下，OPD 的吸收率下降有一个简洁解释：

$$
\text{剩余误差}
\longrightarrow
\text{按曲率分解的弛豫模式}
\longrightarrow
\text{高曲率先消耗，低曲率接管尾部}.
$$

学习率折叠、数据规模不敏感和冻结状态后仍能持续训练，均与该解释一致。其核心预测是 $v(t)\simeq\eta\langle\lambda\rangle_t$，以及 $d\langle\lambda\rangle_t/dt=-\eta\operatorname{Var}(\lambda)\le0$。

这个模型应被视为一个基线理论，而不是完整的 OPD 机制。它不能单独决定谱为何具有某种形状，也不能忽略 Adam 的时变预条件、on-policy 状态漂移、曲率锐化和距离定义差异。最稳妥的表述是：**OPD 的长尾首先是参数空间谱宽的表现；观测曲线偏离该基线的部分，才需要由优化器、噪声和状态分布的时变性解释。**

### 参考文献

[^fu26]: Z. Fu, B. He, Y. Zuo, et al. _Rethinking On-Policy Distillation of Large Language Models II: One Training Example_. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^amari98]: S. Amari. _Natural Gradient Works Efficiently in Learning_. Neural Computation 10(2):251--276, 1998.

[^amari00]: S. Amari, S.-i. Nagaoka. _Methods of Information Geometry_. AMS/Oxford, 2000.

[^martens14]: J. Martens. _New Insights and Perspectives on the Natural Gradient Method_. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^jacot18]: A. Jacot, F. Gabriel, C. Hongler. _Neural Tangent Kernel: Convergence and Generalization in Neural Networks_. NeurIPS 2018. [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)

[^arora19]: S. Arora, S. S. Du, W. Hu, Z. Li, R. Salakhutdinov, R. Wang. _Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks_. ICML 2019. [arXiv:1901.08584](https://arxiv.org/abs/1901.08584)

[^chizat19]: L. Chizat, E. Oyallon, F. Bach. _On Lazy Training in Differentiable Programming_. NeurIPS 2019.

[^hu21]: W. Hu, L. Xiao, B. Adlam, J. Pennington. _The Surprising Simplicity of the Early-Time Learning Dynamics of Neural Networks_. NeurIPS 2021. [arXiv:2006.14539](https://arxiv.org/abs/2006.14539)

[^belfer24]: Y. Belfer, A. Geifman, M. Galun, R. Basri. _Spectral Analysis of the Neural Tangent Kernel for Deep Residual Networks_. JMLR 25(184):1--49, 2024. [jmlr.org/papers/v25/22-0597.html](https://jmlr.org/papers/v25/22-0597.html)

[^cai26]: Y. Cai, D. Cao, L. Lin, et al. _Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation_. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^bordelon20]: A. Bordelon, A. Canatar, C. Pehlevan. _Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks_. ICML 2020. [arXiv:2002.02561](https://arxiv.org/abs/2002.02561)

[^yao07]: Y. Yao, L. Rosasco, A. Caponnetto. _On Early Stopping in Gradient Descent Learning_. Constructive Approximation 26:289--315, 2007.

[^dieuleveut16]: A. Dieuleveut, F. Bach. _Nonparametric Stochastic Approximation with Large Step-Sizes_. Annals of Statistics 44(4):1363--1399, 2016.

[^mandt17]: S. Mandt, M. Hoffman, D. Blei. _Stochastic Gradient Descent as Approximate Bayesian Inference_. JMLR 18(134):1--35, 2017. [arXiv:1704.04289](https://arxiv.org/abs/1704.04289)

[^kingma15]: D. Kingma, J. Ba. _Adam: A Method for Stochastic Optimization_. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

[^balles18]: L. Balles, P. Hennig. _Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients_. ICML 2018. [arXiv:1705.07774](https://arxiv.org/abs/1705.07774)

[^cohen21]: J. Cohen, A. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar. _Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability_. ICLR 2021. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
