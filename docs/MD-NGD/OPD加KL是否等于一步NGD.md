# 用 Natural Gradient 做 OPD：MDPO 形式究竟何时成立？

## 结论

设第 $k$ 轮 rollout 策略为 $\pi_k=\pi_{\theta_k}$，教师为固定策略 $q=\pi^\star$。在冻结的 on-policy 上下文分布 $d_k$ 上定义

$$
L_k(\theta)
=
\mathbb E_{c\sim d_k}
\left[
D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid c)\Vert q(\cdot\mid c)\bigr)
\right].
$$

令

$$
g_k=\nabla_\theta L_k(\theta_k),
\qquad
F_k
=
\mathbb E_{c\sim d_k,\,a\sim\pi_k(\cdot\mid c)}
\left[
s_k(c,a)s_k(c,a)^\top
\right],
$$

其中 $s_k(c,a)=\nabla_\theta\log\pi_{\theta_k}(a\mid c)$。阻尼 Natural Gradient（NG）一步为

$$
\boxed{
\theta_{k+1}
=
\theta_k-\eta(F_k+\lambda I)^{-1}g_k
}
$$

（无阻尼且 $F_k$ 奇异时用 Moore--Penrose 伪逆）。

可以使用 MDPO 风格的 KL 子问题来表达这一步，但必须使用 **OPD 在 $\theta_k$ 处的一阶线性化**：

$$
\boxed{
\min_\theta
g_k^\top(\theta-\theta_k)
+
\frac{1}{\eta}
\mathbb E_{c\sim d_k}
D_{\mathrm{KL}}
\bigl(\pi_\theta(\cdot\mid c)\Vert\pi_k(\cdot\mid c)\bigr)
}
\tag{MD-OPD}
$$

在 $\theta_k$ 附近把 KL 二次化并精确解该局部问题，才得到上面的参数空间 NG 一步。下面两种做法都不与 NG 一步严格等价：

1. 把线性项换回完整的 $L_k(\theta)$；这得到的是 proximal-point 更新，不是 mirror-descent / NG 更新。
2. 从 $\theta_k$ 出发，用有限步普通 SGD 求解 (MD-OPD)；第一步甚至只是普通梯度，因为锚定 KL 在起点的梯度为零。

因此最短的回答是：**可以借用 MDPO 的目标形式，但不能直接照搬 MDPO 的普通 SGD 实现并称之为 Natural Gradient。**

## 1. 先固定 OPD 的外层分布

序列生成时，上下文 $c=(x,y_{<t})$ 的分布依赖学生策略。完整 on-policy OPD 可写成

$$
\mathcal L(\theta)
=
\mathbb E_{c\sim d_\theta}
\left[
D_{\mathrm{KL}}(\pi_\theta(\cdot\mid c)\Vert q(\cdot\mid c))
\right].
$$

每轮先由旧策略 $\pi_k$ rollout，再在同一批上下文上做若干次更新时，实际优化的是冻结分布的 surrogate $L_k(\theta)$，而不是对 $d_\theta$ 的依赖做完全求导。这里与 MDPO 冻结旧策略 occupancy 的做法相同。

本文讨论的 $g_k$、$F_k$ 和所有 KL 都对同一个 $d_k$ 取期望。否则，所谓“该梯度对应哪个 Fisher”并不明确。

## 2. OPD 梯度的精确形式

先看固定上下文 $c$。记

$$
p_a=\pi_\theta(a\mid c),
\qquad
q_a=\pi^\star(a\mid c),
\qquad
\ell_a=\log\frac{p_a}{q_a},
$$

则

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_a p_a\ell_a.
$$

利用 $\nabla_\theta p_a=p_a\nabla_\theta\log p_a$ 以及 $\sum_a\nabla_\theta p_a=0$，有

$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(p\Vert q)
&=
\sum_a \nabla_\theta p_a(\ell_a+1) \\
&=
\sum_a p_a\nabla_\theta\log p_a\,\ell_a \\
&=
\mathbb E_{a\sim p}
\left[
\nabla_\theta\log p_a
\log\frac{p_a}{q_a}
\right].
\end{aligned}
$$

因此

$$
\boxed{
g_k
=
\mathbb E_{c\sim d_k,\,a\sim\pi_k}
\left[
s_k(c,a)
\log\frac{\pi_k(a\mid c)}{q(a\mid c)}
\right]
}
\tag{1}
$$

可以减去任何只依赖 $c$ 的 baseline。一个自然选择是

$$
b_k(c)=D_{\mathrm{KL}}(\pi_k(\cdot\mid c)\Vert q(\cdot\mid c)),
$$

于是

$$
g_k
=
\mathbb E_{c,a}
\left[
s_k(c,a)
\left(
\log\frac{\pi_k(a\mid c)}{q(a\mid c)}-b_k(c)
\right)
\right].
$$

这看起来像 policy gradient，只是“优势”由教师与学生的 log-density ratio 给出。由于我们在最小化 KL，更新方向是 $-g_k$。

## 3. Natural Gradient 从哪里来

考虑局部信赖域问题

$$
\min_\delta
g_k^\top\delta
\quad\text{s.t.}\quad
\mathbb E_{c\sim d_k}
D_{\mathrm{KL}}(\pi_k(\cdot\mid c)\Vert\pi_{\theta_k+\delta}(\cdot\mid c))
\le \varepsilon.
$$

在 $\delta=0$ 处，任一方向的策略 KL 都满足

$$
\mathbb E_c D_{\mathrm{KL}}(\pi_k\Vert\pi_{\theta_k+\delta})
=
\frac12\delta^\top F_k\delta
+O(\Vert\delta\Vert^3).
$$

反向放置 KL 也有相同的二阶项：

$$
\mathbb E_c D_{\mathrm{KL}}(\pi_{\theta_k+\delta}\Vert\pi_k)
=
\frac12\delta^\top F_k\delta
+O(\Vert\delta\Vert^3).
$$

所以 KL 方向只在三阶及以上影响有限步更新。相应的 penalty 子问题为

$$
\min_\delta
g_k^\top\delta
+\frac{1}{2\eta}\delta^\top F_k\delta.
$$

一阶最优条件是

$$
g_k+\frac1\eta F_k\delta=0,
$$

从而

$$
\boxed{
\delta_{\mathrm{NG}}=-\eta F_k^\dagger g_k.
}
\tag{2}
$$

这说明 Natural Gradient 是“目标一阶化、KL 二阶化”后的局部解。它并不要求 OPD 损失自身的 Hessian 等于 Fisher。

## 4. MDPO 风格子问题与 NG 的关系

把二次 KL 恢复为精确的锚定 KL，就得到 (MD-OPD)。这与 Mirror Descent Policy Optimization（MDPO）[^tomar20] 的“冻结旧策略、线性化任务目标、再加 KL proximity”结构相同：

$$
M_k(\theta)
=
g_k^\top(\theta-\theta_k)
+\frac1\eta
\mathbb E_{c\sim d_k}
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_k).
$$

在 $\theta_k$ 处有

$$
\nabla M_k(\theta_k)=g_k,
\qquad
\nabla^2 M_k(\theta_k)=\frac1\eta F_k.
$$

故 $M_k$ 在起点的一步 Newton 解恰为

$$
-\left(\frac1\eta F_k\right)^\dagger g_k
=
-\eta F_k^\dagger g_k.
$$

这就是 MDPO 形式与参数空间 NG 相交的准确位置：

$$
\boxed{
\text{线性化 OPD}
+
\text{局部二次化策略 KL}
+
\text{精确解局部模型}
=
\text{一步参数空间 NG}
}
$$

但精确求解非线性的 $M_k(\theta)$，一般也不严格等于式 (2)。原因是神经网络参数 $\theta$ 通常不是 categorical 分布族的自然坐标，而且 $\theta\mapsto\pi_\theta$ 的像不是完整的概率单纯形。二者只在局部小步意义下吻合。

### 为什么 MDPO 的有限步普通 SGD 不是 NG

若内层从 $\theta_k^{(0)}=\theta_k$ 开始做欧氏 SGD，

$$
\theta_k^{(i+1)}
=
\theta_k^{(i)}-\alpha\nabla M_k(\theta_k^{(i)}),
$$

则第一步满足

$$
\nabla_\theta
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_k)
\big|_{\theta=\theta_k}
=0,
$$

所以

$$
\theta_k^{(1)}-\theta_k=-\alpha g_k.
$$

它是 vanilla gradient，不是 $-\eta F_k^\dagger g_k$。之后的多步 SGD 只是利用 KL 的曲率逐渐逼近子问题解；有限步轨迹依赖内层学习率、步数和 $F_k$ 的条件数。把这种实现称为“隐式 Natural Gradient”并不准确。

## 5. 完整 OPD 加锚定 KL 是另一个算法

现在考虑常见但不同的目标

$$
P_k(\theta)
=
L_k(\theta)
+
\frac1\eta
\mathbb E_{c\sim d_k}
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_k).
\tag{3}
$$

这是 KL-proximal point，而不是 mirror descent，因为第一项没有在 $\theta_k$ 处线性化。记

$$
H_k=\nabla^2 L_k(\theta_k).
$$

在 $\theta_k$ 附近，

$$
P_k(\theta_k+\delta)
=
\mathrm{const}
+g_k^\top\delta
+\frac12\delta^\top H_k\delta
+\frac{1}{2\eta}\delta^\top F_k\delta
+O(\Vert\delta\Vert^3).
$$

因此局部最优位移为

$$
\boxed{
\delta_{\mathrm{prox}}
=
-\left(H_k+\frac1\eta F_k\right)^\dagger g_k.
}
\tag{4}
$$

而不是一般意义下的 $-\eta F_k^\dagger g_k$。

只有额外假设 $H_k\approx F_k$ 时，才有

$$
\delta_{\mathrm{prox}}
\approx
-\frac{\eta}{1+\eta}F_k^\dagger g_k.
\tag{5}
$$

这里必须强调：$H_k\approx F_k$ 不是任意 OPD 状态下的恒等式。它在学生已接近教师时成立到主导阶，或作为 Fisher / Gauss--Newton 近似被人为采用。远离教师时，OPD 的真实 Hessian 含有额外的 residual-curvature 项，甚至不必半正定。

所以原先“完整 OPD 加 KL 等价于步长 $\eta/(1+\eta)$ 的 NG”应改成：**只在局部二次模型同时采用 $H_k\approx F_k$ 时近似如此。**

## 6. 固定上下文、自由分布空间中的精确解

参数空间容易遮蔽结构。对一个固定上下文，暂时把候选分布 $p$ 当成单纯形中的自由变量，旧策略为 $\mu$，教师为 $q$。

### 6.1 完整 OPD proximal 目标

考虑

$$
\min_{p\in\Delta}
D_{\mathrm{KL}}(p\Vert q)
+\frac1\eta D_{\mathrm{KL}}(p\Vert\mu).
$$

加入约束 $\sum_a p_a=1$ 的乘子 $\lambda$：

$$
\mathcal J(p,\lambda)
=
\sum_a p_a\log\frac{p_a}{q_a}
+\frac1\eta\sum_a p_a\log\frac{p_a}{\mu_a}
+\lambda\left(\sum_a p_a-1\right).
$$

一阶条件为

$$
\left(1+\frac1\eta\right)(\log p_a+1)
-\log q_a
-\frac1\eta\log\mu_a
+\lambda
=0.
$$

消去与 $a$ 无关的常数后，

$$
\boxed{
p_{k+1}(a)
=
\frac{
\mu_a^{1/(1+\eta)}q_a^{\eta/(1+\eta)}
}{
\sum_b\mu_b^{1/(1+\eta)}q_b^{\eta/(1+\eta)}
}.
}
\tag{6}
$$

即在 log-probability / logit 坐标中，从旧策略向教师移动比例 $\eta/(1+\eta)$。这是完整 proximal 问题的精确函数空间解。

### 6.2 线性化 OPD 的 KL-MD 目标

在 $p=\mu$ 处，$D_{\mathrm{KL}}(p\Vert q)$ 关于 $p$ 的梯度是

$$
h_a=\log\frac{\mu_a}{q_a}+1.
$$

常数 $1$ 在单纯形约束下可以忽略。KL-MD 子问题是

$$
\min_{p\in\Delta}
\sum_a p_a h_a
+\frac1\eta D_{\mathrm{KL}}(p\Vert\mu).
$$

一阶条件给出指数梯度更新

$$
p_{k+1}(a)
\propto
\mu_a\exp(-\eta h_a).
$$

代入 $h_a$ 后得到

$$
\boxed{
p_{k+1}(a)
=
\frac{
\mu_a^{1-\eta}q_a^\eta
}{
\sum_b\mu_b^{1-\eta}q_b^\eta
}.
}
\tag{7}
$$

这才是自由分布空间中，MDPO 风格的精确 KL-MD 更新。两者比较：

| 更新                        | 教师在 log 空间中的权重 |
| --------------------------- | ----------------------: |
| 完整 OPD proximal，式 (6)   |         $\eta/(1+\eta)$ |
| 线性化 OPD 的 KL-MD，式 (7) |                  $\eta$ |

当 $\eta\to0$ 时，

$$
\frac{\eta}{1+\eta}=\eta-\eta^2+O(\eta^3),
$$

所以二者只在一阶小步意义下相同。$\eta=1$ 时差异尤其明显：KL-MD 一步直接给出 $p_{k+1}=q$，而完整 proximal 只给出归一化几何平均 $p_{k+1}\propto\sqrt{\mu q}$。

式 (7) 是分布空间的有限 mirror step，不应与参数空间的线性切向量 $-\eta F_k^\dagger g_k$ 逐字等同。后者是它在 $\eta\to0$ 时经 Fisher 几何拉回参数空间所得的局部速度；有限步且有神经网络共享参数时，一般无法精确实现每个上下文各自的式 (7)。

## 7. 为什么“OPD Hessian 就是 Fisher”不精确

令固定上下文的学生 logits 为 $z$，$p=\mathrm{softmax}(z)$，教师分布为 $q$，并定义

$$
C(p)=\operatorname{Diag}(p)-pp^\top,
\qquad
r=\log p-\log q,
\qquad
\bar r=p^\top r=D_{\mathrm{KL}}(p\Vert q).
$$

对反向 KL

$$
\ell(z)=D_{\mathrm{KL}}(p_z\Vert q),
$$

其 logit 梯度是

$$
\boxed{
\nabla_z\ell=C(p)r.
}
\tag{8}
$$

分量形式为

$$
[\nabla_z\ell]_i=p_i(r_i-\bar r).
$$

再次求导得到精确 Hessian

$$
\boxed{
[\nabla_z^2\ell]_{ij}
=
p_i(\delta_{ij}-p_j)(r_i-\bar r+1)
-p_i p_j(r_j-\bar r).
}
\tag{9}
$$

当 $p=q$ 时，$r=0$ 且 $\bar r=0$，式 (9) 才退化为

$$
\nabla_z^2\ell\big|_{p=q}=C(q),
$$

也就是 categorical Fisher。远离教师时，式 (9) 还含依赖 $r$ 的项。因此

$$
\nabla_z^2D_{\mathrm{KL}}(p_z\Vert q)=C(p)
$$

不是普遍恒等式，只在 $p=q$ 处成立。

若 logits 由神经网络 $z_\theta(c)$ 给出，记 $J_c=\partial z_\theta(c)/\partial\theta$。参数 Hessian 还多一项 logits 的二阶导数：

$$
\nabla_\theta^2\ell
=
J_c^\top(\nabla_z^2\ell)J_c
+
\sum_i[\nabla_z\ell]_i\nabla_\theta^2 z_i.
\tag{10}
$$

而参数 Fisher 是

$$
F_k
=
\mathbb E_c[J_c^\top C(\pi_k)J_c].
\tag{11}
$$

故 $H_k=F_k$ 需要同时压掉式 (9) 中的 residual-curvature 和式 (10) 中的模型二阶项。学生接近教师时这些残差项趋近于零；采用线性化 logits 加 Gauss--Newton 近似时也会主动忽略它们。这就是 Cai 等（2026）附录 F.5[^cai26] 中 $A=\mathbb E[J_c^\top C(p_k)J_c]$ 的正确适用边界。

还要注意 softmax logits 有加常数不变性，因此 $C(p)$ 奇异。类似 $z^\star-z_k$ 的 logit 差只有在选定 gauge 后才唯一；更稳妥的对象是 log-probability ratio、中心化 logits，或直接使用 Fisher 伪逆。

## 8. 实际该怎样用 Natural Gradient 做 OPD

如果目标确实是“每轮做一步 OPD Natural Gradient”，最直接的算法不是用很多步普通 SGD 模拟 MDPO，而是显式求解 Fisher 线性系统。

第 $k$ 轮：

1. 用 $\pi_k$ rollout，得到上下文批次 $c\sim d_k$，在本轮内冻结这些上下文。
2. 用式 (1) 估计 OPD 梯度 $\hat g_k$。
3. 用同一上下文和学生分布构造 empirical Fisher / Fisher-vector product：

$$
\widehat F_k v
=
\frac1N\sum_{n=1}^N
J_{c_n}^\top C(\pi_k(\cdot\mid c_n))J_{c_n}v.
$$

4. 用共轭梯度等方法求

$$
(\widehat F_k+\lambda I)v_k=\hat g_k.
$$

5. 令候选更新为

$$
\theta_{k+1}=\theta_k-\alpha v_k,
$$

并通过 line search 或目标 KL 控制实际步长：

$$
\mathbb E_{c\sim d_k}
D_{\mathrm{KL}}(\pi_k\Vert\pi_{k+1})
\le\delta.
$$

二次模型下，若希望直接缩放到 KL 半径 $\delta$，可取

$$
\alpha
=
\sqrt{
\frac{2\delta}{v_k^\top\widehat F_k v_k}
}.
$$

若不想显式求解 Fisher，也可以实现 MDPO 风格的 (MD-OPD)，但应准确描述为“用多步 SGD 近似求解 KL-MD 子问题”，并监测：

- inner optimality residual 是否足够小；
- 实际策略 KL 是否落在期望半径内；
- inner loop 是否因 Fisher 病态而只推进了高曲率方向；
- 参数化策略是否能表示式 (7) 所要求的函数空间更新。

若直接优化式 (3)，则算法应称为 **KL-proximal OPD**。它可以是合理且更保守的训练方法，但不应称为“一步长为 $\eta$ 的 Natural Gradient”。

## 9. 最终关系图

$$
\boxed{
\begin{aligned}
\text{OPD NG 一步}
&:\quad -\eta F_k^\dagger g_k, \\
\text{局部 MD-OPD}
&:\quad
\arg\min_\delta
\left(g_k^\top\delta+\frac{1}{2\eta}\delta^\top F_k\delta\right), \\
\text{精确函数空间 KL-MD}
&:\quad
p_{k+1}\propto\mu^{1-\eta}q^\eta, \\
\text{完整 KL-proximal OPD}
&:\quad
p_{k+1}\propto\mu^{1/(1+\eta)}q^{\eta/(1+\eta)}, \\
\text{MDPO 式有限步欧氏 SGD}
&:\quad
\text{上述子问题的数值近似路径，不是 NG 本身。}
\end{aligned}
}
$$

因此，用 MDPO 形式做 OPD 是可以的，但必须先回答你想实现的是哪一个对象：

- 想要严格的局部 Natural Gradient：计算 $F_k^\dagger g_k$，或精确求解其局部二次子问题。
- 想要函数空间的 KL-Mirror Descent：线性化 OPD，并求解 (MD-OPD)。
- 想要保守地逼近教师：完整 OPD 加锚定 KL 很自然，但它是 proximal OPD。
- 只是在 MDPO surrogate 上跑若干步 Adam / SGD：这是实用近似，不能仅凭目标里出现 KL 就宣称等价于 NG。

## 10. rollout 后多轮 PPO epoch：实际在优化什么

本节专门讨论如下训练循环。第 $k$ 轮先冻结 rollout 策略

$$
\mu=\pi_{\theta_k},
$$

用它生成上下文，然后在同一批上下文上进行 $m$ 个 PPO-style epoch。每个 epoch 的候选策略记为 $\pi_\theta$；内层完成后令

$$
\mu\leftarrow\pi_{\theta_{k+1}}.
$$

这里不讨论 Importance Sampling。它只是在旧策略产生的样本上无偏或近似无偏地估计候选策略期望的手段；它不改变所优化函数的几何，也不把欧氏 SGD 自动变成 Natural Gradient。

### 10.1 冻结 rollout 后的完整目标

令 $d_\mu$ 表示该轮 rollout 所诱导的上下文分布，教师为 $q=\pi^\star$。忽略采样估计误差，完整 OPD 项为

$$
L_\mu(\theta)
=
\mathbb E_{c\sim d_\mu}
\left[
D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid c)\Vert q(\cdot\mid c)\bigr)
\right].
$$

若在每个 epoch 都使用当前 $\pi_\theta$ 重新计算这项，并同时加锚定 KL，内层优化的精确对象是

$$
\boxed{
P_\mu(\theta)
=
L_\mu(\theta)
+
\frac1\eta
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}
\bigl(\pi_\theta(\cdot\mid c)\Vert\mu(\cdot\mid c)\bigr).
}
\tag{12}
$$

这是一个以 $\mu$ 为中心的 KL-proximal-point 子问题。它与 TRPO 的“固定旧策略分布、约束候选策略变化”的局部信赖域思想相近[^schulman15]，但这里使用的是软 penalty，且任务项是 OPD 的完整反向 KL。

它不是严格 MD-OPD，因为 mirror descent 的第一项应是旧点处的**一阶模型**，而式 (12) 仍保留 $L_\mu(\theta)$ 的非线性、非二次曲率。这个区别不是术语问题：两种目标的精确函数空间解不同，见第 6 节。

### 10.2 为什么 MD 要在分布空间一阶线性化

对固定上下文 $c$，先把策略分布 $p$ 看作单纯形 $\Delta$ 中的自由变量，并设旧分布为 $\mu$。定义

$$
f_c(p)=D_{\mathrm{KL}}(p\Vert q)
=
\sum_a p_a\log\frac{p_a}{q_a}.
$$

分量导数是

$$
\frac{\partial f_c}{\partial p_a}
=
\log\frac{p_a}{q_a}+1.
$$

因此在 $p=\mu$ 处的一阶 Taylor 展开为

$$
\begin{aligned}
f_c(p)
&=
f_c(\mu)
+
\sum_a(p_a-\mu_a)
\left(
\log\frac{\mu_a}{q_a}+1
\right)
+O(\Vert p-\mu\Vert_2^2) \\
&=
\mathrm{const}
+
\sum_a p_a\log\frac{\mu_a}{q_a}
+O(\Vert p-\mu\Vert_2^2).
\end{aligned}
\tag{13}
$$

第二个等号利用了单纯形切空间条件

$$
\sum_a(p_a-\mu_a)=0,
$$

所以常数 $1$ 对可行方向没有贡献。令

$$
h_\mu(c,a)=\log\frac{\mu(a\mid c)}{q(a\mid c)}.
$$

则冻结 rollout 后 OPD 的**分布空间一阶模型**为

$$
\widetilde L_\mu(\pi)
=
\mathbb E_{c\sim d_\mu}
\left[
\sum_a\pi(a\mid c)h_\mu(c,a)
\right]
+\mathrm{const}.
\tag{14}
$$

也可以使用中心化版本

$$
A_\mu(c,a)
=
h_\mu(c,a)
-D_{\mathrm{KL}}\bigl(\mu(\cdot\mid c)\Vert q(\cdot\mid c)\bigr),
$$

因为对每个 $c$ 都有

$$
\sum_a\pi(a\mid c)A_\mu(c,a)
=
\sum_a\pi(a\mid c)h_\mu(c,a)
-D_{\mathrm{KL}}(\mu\Vert q),
$$

两者只相差不依赖候选策略的常数。它是 OPD 对应的固定 advantage / cost。

之所以在分布 $p$ 而不是神经网络参数 $\theta$ 上进行这一步，是因为 KL mirror map 的定义域是概率单纯形：负熵

$$
\psi(p)=\sum_a p_a\log p_a
$$

所诱导的 Bregman divergence 正是

$$
D_\psi(p,\mu)=D_{\mathrm{KL}}(p\Vert\mu).
$$

因此 mirror descent 的“梯度线性项”是 $f_c$ 对 $p$ 的一阶变分，即式 (13)，而不是对任意神经网络坐标 $\theta$ 的 Taylor 展开。神经网络参数化只是在后续把可行的函数空间策略限制为 $\{\pi_\theta\}$；一般不能把 $\theta$ 当作单纯形的仿射坐标或 softmax 指数族的自然坐标。MDPO[^tomar20] 也是先从策略空间的 mirror-descent / trust-region 子问题出发，再用参数化策略上的多步梯度法近似求解。

据此，严格的函数空间 KL-MD-OPD 子问题是

$$
\boxed{
\min_\pi
\mathbb E_{c\sim d_\mu}
\left[
\sum_a\pi(a\mid c)h_\mu(c,a)
+\frac1\eta
D_{\mathrm{KL}}\bigl(\pi(\cdot\mid c)\Vert\mu(\cdot\mid c)\bigr)
\right].
}
\tag{15}
$$

这里“线性化”不是说算法做完后只能走一条参数直线；它是指把本轮任务损失冻结为旧策略处的切平面。KL 项则不线性化，因为它定义了本轮允许如何移动的几何。

### 10.3 线性化的精确 mirror step 如何得到

对一个固定 $c$，为式 (15) 加入约束 $\sum_a p_a=1$ 的乘子 $\lambda$，有

$$
\mathcal G(p,\lambda)
=
\sum_a p_a h_\mu(a)
+\frac1\eta\sum_a p_a\log\frac{p_a}{\mu_a}
+\lambda\left(\sum_a p_a-1\right).
$$

对 $p_a$ 求导并令其为零：

$$
h_\mu(a)
+\frac1\eta\left(\log\frac{p_a}{\mu_a}+1\right)
+\lambda
=0.
$$

把与 $a$ 无关的项吸收到归一化常数中，得到

$$
p_a\propto\mu_a\exp\bigl(-\eta h_\mu(a)\bigr).
$$

代入 $h_\mu(a)=\log(\mu_a/q_a)$：

$$
\boxed{
p_a
=
\frac{
\mu_a^{1-\eta}q_a^\eta
}{
\sum_b\mu_b^{1-\eta}q_b^\eta
}.
}
\tag{16}
$$

这解释了“为什么要线性化”：它使任务项在策略分布中变成线性函数，从而与 KL mirror map 组成一个可精确求解的 mirror-proximal 子问题。相反，若保留 $D_{\mathrm{KL}}(p\Vert q)$，得到的是另一个 proximal-point 问题，其解为第 6 节的几何混合，而非式 (16)。

### 10.4 你的多 epoch SGD 到底是什么 GD

从式 (12) 出发，若 inner optimizer 是 SGD，准确的更新就是

$$
\theta^{(j+1)}
=
\theta^{(j)}
-\alpha_j\widehat{\nabla}_\theta P_\mu(\theta^{(j)}).
\tag{17}
$$

这里 $\widehat{\nabla}$ 表示 minibatch 估计。故它是：**在参数空间上，对冻结-rollout 的 KL-proximal OPD 目标做随机欧氏梯度下降。** 若实际使用 Adam，则更准确地说是带自适应对角预条件器的随机一阶法，而非严格欧氏 SGD。

为什么第一步不带 Natural Gradient 预条件？令

$$
g=\nabla_\theta L_\mu(\theta_k).
$$

由于锚定策略就是初始策略，

$$
\nabla_\theta
D_{\mathrm{KL}}(\pi_\theta\Vert\mu)
\big|_{\theta=\theta_k}=0.
$$

所以

$$
\theta^{(1)}-\theta_k=-\alpha_0g.
\tag{18}
$$

它是 vanilla OPD gradient step。后续 epoch 的 KL 梯度会把参数拉回 $\mu$ 附近，但 SGD 从未显式求解 $F^{-1}g$，所以这条内层轨迹不是 Natural Gradient 轨迹。

为看清局部动力学，置

$$
\delta_j=\theta^{(j)}-\theta_k,
\qquad
H=\nabla_\theta^2L_\mu(\theta_k),
\qquad
F=F(\theta_k).
$$

在 $\delta=0$ 附近，KL 的一阶项为零、二阶项为 $F$：

$$
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}(\pi_{\theta_k+\delta}\Vert\mu)
=
\frac12\delta^\top F\delta+O(\Vert\delta\Vert^3).
$$

故

$$
P_\mu(\theta_k+\delta)
=
\mathrm{const}
+g^\top\delta
+\frac12\delta^\top
\left(H+\frac1\eta F\right)
\delta
+O(\Vert\delta\Vert^3).
\tag{19}
$$

取固定内层学习率 $\alpha$ 并暂时忽略采样噪声，式 (17) 的线性化递推为

$$
\boxed{
\delta_{j+1}
=
\left[I-\alpha\left(H+\frac1\eta F\right)\right]\delta_j
-\alpha g.
}
\tag{20}
$$

这就是你的多 epoch 动力学。它是欧氏 GD 对曲率

$$
B=H+\frac1\eta F
$$

的线性系统的迭代，而不是 F.5 中每一步左乘 $F^{-1}$ 的 NGD。

若 $B$ 对相关子空间可逆，并且 $0<\alpha<2/\lambda_{\max}(B)$，从 $\delta_0=0$ 出发有

$$
\begin{aligned}
\delta_m
&=
-\alpha\sum_{r=0}^{m-1}(I-\alpha B)^r g \\
&=
-\left[I-(I-\alpha B)^m\right]B^{-1}g.
\end{aligned}
\tag{21}
$$

因此 $m\to\infty$ 时的 proximal 点是

$$
\boxed{
\delta_\infty
=
-\left(H+\frac1\eta F\right)^{-1}g.
}
\tag{22}
$$

式 (22) 也可直接由式 (19) 对 $\delta$ 求导得到：

$$
g+\left(H+\frac1\eta F\right)\delta=0.
$$

这就是这里“Newton 解”的来源：二次模型的 Hessian 恰为 $B$，Newton 法对一个严格二次函数一步便解出其驻点；它不是说普通 SGD 做了一步 Newton。有限 $m$ 时你走的是式 (21)，只有内层充分收敛时才近似式 (22)。

### 10.5 何时才复现 F.5 的 NGD 动力学

Cai 等（2026）[^cai26] 的 F.5 在固定局部几何下使用

$$
L(\Delta)
=
\frac12\Delta^\top A\Delta-b^\top\Delta+\mathrm{const},
\qquad
F=A.
\tag{23}
$$

其中 $\Delta=\theta-\theta_0$。对应的显式 NGD 更新是

$$
\Delta_{k+1}
=
\Delta_k
-\gamma A^\dagger(A\Delta_k-b)
=
(1-\gamma)\Delta_k+\gamma A^\dagger b.
\tag{24}
$$

现在把外层 rollout 策略视为当前 $\Delta_k$，并额外作 F.5 的强近似

$$
H=A,
\qquad
F=A.
$$

若每轮 inner epoch 已经精确求解式 (12)，并且式 (23) 的固定局部二次模型成立，则由式 (22)

$$
\begin{aligned}
\Delta_{k+1}-\Delta_k
&=
-\left(A+\frac1\eta A\right)^\dagger(A\Delta_k-b) \\
&=
-\frac{\eta}{1+\eta}A^\dagger(A\Delta_k-b).
\end{aligned}
$$

等价地，写成更清楚的仿射递推：

$$
\boxed{
\Delta_{k+1}
=
\frac1{1+\eta}\Delta_k
+\frac{\eta}{1+\eta}A^\dagger b.
}
\tag{25}
$$

比较式 (24)，可见式 (25) 恰是 F.5 的 NGD 递推，且有效步长为

$$
\boxed{
\gamma=\frac{\eta}{1+\eta},
}
$$

不是 $\eta$。

这个结论需要同时满足以下条件：

1. 内层 epoch 足够多，使 SGD 已近似收敛到 KL-proximal 子问题的解；
2. OPD 的局部 Hessian 可以以 $A$ 近似，且锚定 KL 的 Fisher 也是同一个固定 $A$；
3. rollout 上下文分布在每个 outer iteration 内冻结；
4. 忽略重采样与远离局部邻域后 $A$、$b$、$H$ 的变化。

任何一个条件不满足时，实际训练仍然是式 (17) 的多 epoch 随机一阶过程。尤其在有限 $m$ 时，式 (21) 中的谱滤波因子

$$
I-(I-\alpha B)^m
$$

会保留不同曲率方向不同的收敛速度；这正是欧氏 GD，而不是 F.5 的 NGD 中所有 Fisher 特征方向共享同一收缩因子的动力学。

### 参考文献

[^tomar20]: Manan Tomar, Lior Shani, Yonathan Efroni, Mohammad Ghavamzadeh. Mirror Descent Policy Optimization. arXiv preprint 2020. [arXiv:2005.09814](https://arxiv.org/abs/2005.09814)

[^schulman15]: John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel. Trust Region Policy Optimization. ICML 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)

[^cai26]: Yuchen Cai, Ding Cao, Liang Lin, Chunxi Luo, Xin Xu, Kai Yang, Weijie Liu, Saiyong Yang, Tianxiang Zhao, Guangzhong Sun, Guiquan Liu, Junfeng Fang. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv preprint 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)
