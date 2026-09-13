# Primal、$G$、$\mu$，以及 outer=NGD / inner=SGD

两句结论：

1. $G$ 是 Mirror Descent 的势函数，$\mu=\nabla G(\theta)$ 是它的对偶坐标。MD 的本质是：在 primal 里写 Bregman proximal，在 $\mu$ 里做一次平移，再镜像回来。
2. 「每个 outer step 是一步 NGD，inner SGD 只是在解这个 NGD 目标」——作为**局部二次化后的设计意图**说得通；作为精确 MD 或 MDPO 实现的字面描述，过强。Inner 解的是 proximal 子问题，它的精确解是 dual 坐标上的 NGD，不是你在神经网络 $\theta$ 上直接写的 $F^{-1}g$，除非再做一层线性化。

## 1. Primal、$G$、$\mu$ 分别是什么

Raskutti--Mukherjee 的记号里有两套坐标、一个势函数、一个对偶势函数。

### Primal 坐标 $\theta$

这是你写出 Mirror Descent 问题的那套变量。它可以是：

- 概率向量 $\pi$；
- softmax 的 logits $\eta$；
- 神经网络参数。

“Primal”不是神秘的物理量，就是 **MD 子问题所在的坐标系**。

### 势函数 $G$

$G$ 是定义在 primal 上的严格凸、二阶可微函数。它干两件事。

第一，生成 Bregman 发散：

$$
D_G(\theta,\theta')
=
G(\theta)-G(\theta')-\langle\nabla G(\theta'),\theta-\theta'\rangle.
$$

$D_G$ 就是 MD 里那项“别走太远”的尺子。$G(\theta)=\frac12\Vert\theta\Vert_2^2$ 时，$D_G$ 是欧氏平方距离，MD 退化成普通 GD。$G$ 取负熵时，$D_G$ 变成 KL。

第二，给出镜像映射：

$$
\mu=\nabla G(\theta).
$$

$\mu$ 就是对偶坐标。MD 之所以叫 mirror，就是因为更新发生在 $\mu$ 空间，再经 $\nabla G$ 的反函数映回 $\theta$。

### 对偶坐标 $\mu$ 和对偶势 $H$

$H=G^\ast$ 是 $G$ 的凸共轭：

$$
H(\mu)=\sup_{\theta}\bigl(\langle\theta,\mu\rangle-G(\theta)\bigr).
$$

Legendre 对偶给出互逆的梯度：

$$
\mu=\nabla G(\theta),
\qquad
\theta=\nabla H(\mu),
\qquad
\nabla^2 H(\mu)=\bigl(\nabla^2 G(\theta)\bigr)^{-1}.
$$

于是有两套 Riemannian 流形：

- primal：$(\Theta,\nabla^2 G)$；
- dual：$(\Phi,\nabla^2 H)$，其中 $\Phi$ 是 $\nabla G$ 的像。

Raskutti--Mukherjee 的定理是：

> 在 primal 上做 $G$-Bregman MD，等价于在对偶流形 $(\Phi,\nabla^2 H)$ 上做 Natural Gradient。

不是“在同一套 $\theta$ 坐标里，MD 等于 $F(\theta)^{-1}\nabla f$”。

### 更新到底发生在哪

MD 子问题

$$
\theta_{t+1}
=
\arg\min_\theta
\left\{
\langle\nabla f(\theta_t),\theta\rangle
+
\frac{1}{\alpha}D_G(\theta,\theta_t)
\right\}
$$

的一阶条件是

$$
\nabla G(\theta_{t+1})
=
\nabla G(\theta_t)-\alpha\nabla_\theta f(\theta_t),
$$

也就是

$$
\mu_{t+1}=\mu_t-\alpha\nabla_\theta f(\theta_t).
$$

对偶空间里这是一次**平移**。再用 $\theta=\nabla H(\mu)$ 镜像回来。把 $\nabla_\theta f$ 用链式法则写成 $\nabla_\mu f$，就得到对偶 NGD：

$$
\mu_{t+1}
=
\mu_t
-
\alpha\bigl[\nabla^2 H(\mu_t)\bigr]^{-1}\nabla_\mu f.
$$

如果只把 $\nabla G(\theta_{t+1})$ 线性化：

$$
\nabla G(\theta_{t+1})
\approx
\nabla G(\theta_t)+\nabla^2 G(\theta_t)(\theta_{t+1}-\theta_t),
$$

才会得到 primal 坐标里的一步 NGD / Newton：

$$
\theta_{t+1}-\theta_t
\approx
-\alpha\bigl[\nabla^2 G(\theta_t)\bigr]^{-1}\nabla_\theta f.
$$

因此有两层不同的“等于 NGD”：

| 说法                         | 精确含义                                             |
| ---------------------------- | ---------------------------------------------------- |
| 精确 MD                      | 对偶坐标 $\mu$ 上的精确 NGD                          |
| 线性化 MD                    | primal 坐标上、度量 $\nabla^2 G$ 的一步 NGD          |
| 你在网络参数上写的 $F^{-1}g$ | 又要 $\nabla^2 G$ 或局部 Hessian 恰好等于政策 Fisher |

## 2. 三个具体例子

### 欧氏空间

$$
G(\theta)=\frac12\Vert\theta\Vert_2^2,
\qquad
\mu=\theta,
\qquad
H(\mu)=\frac12\Vert\mu\Vert_2^2.
$$

镜像映射是恒等。MD = GD = NGD。没有新几何。

### Softmax / 策略：primal 取概率

令 primal 为分布 $\pi\in\Delta^{V-1}$，

$$
G(\pi)=\sum_i\pi_i\log\pi_i.
$$

则

$$
D_G(\pi,\pi_k)=D_{\mathrm{KL}}(\pi\Vert\pi_k),
\qquad
\mu=\nabla G(\pi)=\log\pi+1.
$$

$\mu$ 本质上就是 logits（单纯形约束下还要减一个常数）。KL-MD

$$
\pi_{k+1}
=
\arg\min_\pi
\left\{
\langle g,\pi\rangle
+
\frac{1}{\alpha}D_{\mathrm{KL}}(\pi\Vert\pi_k)
\right\}
$$

的闭式解是

$$
\pi_{k+1}(i)
\propto
\pi_k(i)\exp(-\alpha g_i),
$$

也就是：把 $-\alpha g$ 加到 $\log\pi$ 上，再 softmax 回来。欧氏平移发生在 $\mu$ 里，几何上这正是 categorical Fisher--Rao 的对偶坐标。

### Softmax：primal 取 logits

令 primal 为 logits $\eta$，势函数取 log-partition

$$
G(\eta)=\log\sum_j e^{\eta_j}.
$$

则对偶坐标就是均值参数

$$
\mu=\nabla G(\eta)=\mathrm{softmax}(\eta)=\pi,
\qquad
H(\pi)=\sum_i\pi_i\log\pi_i,
$$

对偶度量 $\nabla^2 H(\pi)$ 是单纯形上的 Fisher。Raskutti 表里 Bernoulli 的 $G(\theta)=\log(1+e^\theta)$ 就是这个方向：primal 是自然参数，dual 是 $p$。

两套写法是同一件事的 Legendre 对偶。政策优化文献通常把 primal 取成 $\pi$，KL 写成 $D_{\mathrm{KL}}(\pi\Vert\pi_k)$，镜像空间是 logits。

### 神经网络参数为什么突然不等了

网络是

$$
\theta_{\mathrm{nn}}
\;\longrightarrow\;
\eta=f_{\theta}(c)
\;\longrightarrow\;
\pi=\mathrm{softmax}(\eta).
$$

$D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta_k})$ 是 $\pi$ 上的 Bregman，不是 $\theta_{\mathrm{nn}}$ 上的 Bregman。Raskutti 的 $\mu=\nabla G(\theta_{\mathrm{nn}})$ 在这里没有闭式。把 KL 在 $\theta_k$ 处二次化，Hessian 才变成 pullback Fisher

$$
F(\theta_k)
=
\mathbb{E}\bigl[J^{\top}(\mathrm{Diag}(\pi)-\pi\pi^{\top})J\bigr],
$$

这一步 Newton 才是参数空间 NGD。F.5 里的 $A$ 就是这个东西。

## 3. 你的 outer=NGD、inner=SGD 对不对

短答案：**局部设计意图对，字面等式不对。**

把一层 outer 想成

$$
\min_{\theta}
\underbrace{\langle g,\theta\rangle}_{\text{线性化目标}}
+
\frac{1}{\alpha}
\underbrace{D(\theta,\theta_k)}_{\text{靠近旧点}}
$$

inner SGD 是在解这个子问题。若子问题被**精确**解掉，outer 得到的是一步 **MD / proximal**。再分三种鉴定：

1. 子问题是 $\pi$ 或 logits 上的精确 KL-MD：outer $=$ 对偶坐标上的精确 NGD。Inner SGD 只是这个 proximal 的求解器。
2. 子问题在 $\theta_k$ 处被换成二次型 $\frac12(\theta-\theta_k)^{\top}F(\theta-\theta_k)$：精确解就是一步参数 NGD，$\theta-\theta_k=\alpha F^{-1}g$。这时「outer 是 NGD，inner 在解 NGD 目标」完全正确。TRPO 的共轭梯度就是在解这个线性方程，而不是用 SGD。
3. 子问题是 MDPO 那种网络参数上的完整 $\Psi(\theta,\theta_k)$，inner 用欧氏 SGD 走 $m$ 步：outer 是**不精确的 KL proximal**。它和 $F^{-1}g$ 只在 $\theta_k$ 附近一阶重合。$m=1$ 时 KL 的梯度还是 $0$，outer 直接变成 vanilla PG，连 NGD 的边都沾不上。

所以更准确的句子是：

> Inner loop 近似求解 MD 子问题；该子问题的**精确解**是 dual NGD；该子问题的**局部二次解**是参数空间一步 NGD。Inner 的每一步 SGD 本身不是 NGD。

Inner SGD 走的几何仍是 $\Vert\cdot\Vert_2$。它只是一个求解器。你不能因为目标的 Hessian 接近 $F$，就说这些 SGD 步“是”自然梯度。它们最多是在朝 NGD 那个点爬。

还有一个实践上的刺：若 inner 目标真是

$$
\min_v
\frac12 v^{\top}Fv-g^{\top}v,
$$

欧氏 SGD 的条件数就是 $F$ 的条件数，往往很差。这正是人们改用 CG / K-FAC 的原因。用 SGD 去“近似一步 NGD”，理论上能收敛到 NGD 点，但并不是实现 NGD 的好算法。

若 inner 不解二次型，而解完整 KL proximal，outer 其实比线性化 NGD 更强：它是有限步长的镜像步，不是 $F^{-1}g$ 那条切线。这时把 outer 叫 NGD 会把“有限步 proximal”和“一步 Newton”混在一起。

## 4. 和 OPD / F.5 的对齐

F.5 局部目标已经是二次的：

$$
\min_{\Delta\theta}
\frac12\Delta\theta^{\top}A\Delta\theta-b^{\top}\Delta\theta,
\qquad
A=\mathbb{E}_c[J_c^{\top}F_c J_c].
$$

这里 $A$ 就是 $G$ 在 $\theta_0$ 处被二次化后的 $\nabla^2 G$，也是 Fisher。因此：

- 精确解 $\Delta\theta=A^{-1}b$：一步 NGD / Newton；
- 欧氏 GD：沿 $b$ 出发，方向旋转；
- 冻结 $A$ 的 NGD：方向从第一步锁在 $A^{-1}b$。

如果你的算法是「outer 想迈 $A^{-1}b$ 这一步，inner 用 SGD 去解上面这个二次问题」，那就是标准的 inexact Newton / inexact NGD。停止标准不该看欧氏 loss 是否还在降，而该看自然梯度几何里的相对残差

$$
\frac{\Vert A\Delta\theta-b\Vert_{A^{-1}}}{\Vert b\Vert_{A^{-1}}},
$$

这和 [mirror descent.md](/Users/fanghaotian/Desktop/obsidian/opd%20MD-NGD/mirror%20descent.md) 里的说法一致。

如果你的算法是 MDPO 式的「outer 写 KL，inner 欧氏 SGD $m$ 步」，那就不要说 every outer step 是 NGD。它是 inexact MD。只有把 inner 做到接近局部二次的 Newton 点，这个说法才成立。
