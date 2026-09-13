# MDPO 实现的 Mirror Descent 是否等价于 Natural Gradient

结论先说：

> Tomar 等人的 MDPO（[arXiv:2005.09814](https://arxiv.org/abs/2005.09814)）是用欧氏 SGD 去近似求解 KL-Mirror Descent 子问题。它不是参数空间上的 Natural Gradient，也不等于 Raskutti--Mukherjee 意义下“精确 MD = 对偶流形上的 NGD”。
>
> 只有把 MDPO 的局部 surrogate 在 $\theta_k$ 处二次化，并且**精确**解这个二次问题，才会得到一步参数 Fisher 自然梯度。MDPO 的实际算法没有做这件事。

## 先把四个对象分开

后面容易混的是这四件东西：

1. **单纯形上的精确 KL-MD**（tabular / 每状态独立更新）：

    $$
    \pi_{k+1}(\cdot\mid s)
    \propto
    \pi_k(\cdot\mid s)\exp\bigl(t_k A^{\pi_k}(s,\cdot)\bigr).
    $$

    这是指数梯度，也是分布空间里的精确 proximal 步。

2. **Raskutti--Mukherjee 等价**（[arXiv:1310.7780](https://arxiv.org/abs/1310.7780)）：$G$-Bregman Mirror Descent 在 primal 坐标中，等价于对偶 Riemannian 流形 $(\Phi,\nabla^2 H)$ 上的 Natural Gradient。这里的 $H=G^\ast$。

3. **参数空间 Natural Policy Gradient / TRPO**：

    $$
    \theta\leftarrow\theta+\eta F(\theta)^{-1}\nabla_\theta J,
    \qquad
    F(\theta)
    =
    \mathbb{E}_{s,a\sim\pi_\theta}
    \bigl[
    \nabla_\theta\log\pi_\theta(a\mid s)
    \nabla_\theta\log\pi_\theta(a\mid s)^{\top}
    \bigr].
    $$

4. **MDPO 的实现**：外层写一个 KL-MD 风格的 surrogate $\Psi(\theta,\theta_k)$，内层用 $m$ 步**普通 SGD** 去近似最大化它。论文明确说 on-policy MDPO 用 simple SGD，而不是 natural gradient，以避免求 Fisher 逆。

第 1 和 2 在合适坐标下可以等价；第 3 是把 Fisher 拉回神经网络参数；第 4 只是用欧氏梯度去凑第 1 的一个参数化版本。MDPO 属于第 4 类。

## MDPO 实际在做什么

On-policy MDPO 的外层目标是

$$
\theta_{k+1}
\in
\arg\max_{\theta}
\Psi(\theta,\theta_k),
$$

$$
\Psi(\theta,\theta_k)
=
\mathbb{E}_{s\sim\rho_{\theta_k}}
\left[
\mathbb{E}_{a\sim\pi_\theta}
A^{\theta_k}(s,a)
-
\frac{1}{t_k}
D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid s)\Vert\pi_{\theta_k}(\cdot\mid s)\bigr)
\right].
$$

注意三件事实：

- 状态分布冻在旧策略：$s\sim\rho_{\theta_k}$；
- 改进项里的动作仍是候选策略：$a\sim\pi_\theta$；
- KL 是 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta_k})$，与凸优化里 MD 的方向一致，和 TRPO 的 $D_{\mathrm{KL}}(\pi_{\theta_k}\Vert\pi_\theta)$ 相反。

闭式解一般不存在，于是 MDPO 做

$$
\theta_k^{(0)}=\theta_k,
\qquad
\theta_k^{(i+1)}
=
\theta_k^{(i)}+\eta\nabla_\theta\Psi(\theta,\theta_k)\big|_{\theta=\theta_k^{(i)}},
\qquad
\theta_{k+1}=\theta_k^{(m)}.
$$

这是欧氏空间里的多步 SGD，不是

$$
\theta\leftarrow\theta+\eta F(\theta)^{-1}\nabla\Psi.
$$

论文还指出：只做一步时，

$$
\nabla_\theta D_{\mathrm{KL}}\bigl(\pi_\theta\Vert\pi_{\theta_k}\bigr)\big|_{\theta=\theta_k}=0,
$$

于是

$$
\nabla_\theta\Psi(\theta,\theta_k)\big|_{\theta=\theta_k}
=
\mathbb{E}_{s\sim\rho_{\theta_k},\,a\sim\pi_{\theta_k}}
\bigl[
\nabla_\theta\log\pi_{\theta_k}(a\mid s)\,A^{\theta_k}(s,a)
\bigr].
$$

也就是 vanilla policy gradient。KL 信任域在 $m=1$ 时完全没有一阶作用。这是 MDPO 必须 $m>1$ 的原因，也直接说明：**MDPO 的实现路径不是 NGD**。NGD / TRPO 即使只走一步，Fisher 逆仍然在起作用。

内层梯度用旧策略样本时会出现 importance ratio

$$
\frac{\pi_{\theta_k^{(i)}}(a\mid s)}{\pi_{\theta_k}(a\mid s)}.
$$

这是在用 $\pi_{\theta_k}$ 的动作去估计 $a\sim\pi_\theta$ 的期望，和“算不算自然梯度”无关。

Off-policy MDPO 同样是对

$$
\Psi(\theta,\theta_k)
=
\mathbb{E}_{s\sim\mathcal{D}}
\left[
\mathbb{E}_{a\sim\pi_\theta}A^{\theta_k}(s,a)
-
\frac{1}{t_k}
D_{\mathrm{KL}}\bigl(\pi_\theta\Vert\pi_{\theta_k}\bigr)
\right]
$$

做多步 SGD / reparameterization，仍然不乘 $F^{-1}$。

## 精确 MD 何时等于 NGD

Raskutti--Mukherjee 的定理非常窄，也非常精确。令 $G$ 严格凸，$H=G^\ast$，$\mu=\nabla G(\theta)$。Primal 上的 MD

$$
\theta_{t+1}
=
\arg\min_\theta
\left\{
\langle\nabla f_t(\theta_t),\theta\rangle
+
\frac{1}{\alpha_t}D_G(\theta,\theta_t)
\right\}
$$

一阶条件是

$$
\nabla G(\theta_{t+1})
=
\nabla G(\theta_t)-\alpha_t\nabla_\theta f_t(\theta_t).
$$

换到对偶坐标 $\mu$ 后：

$$
\mu_{t+1}
=
\mu_t
-
\alpha_t
\bigl[\nabla^2 H(\mu_t)\bigr]^{-1}
\nabla_\mu f_t.
$$

因此：

> $G$-Bregman MD 在 primal 坐标中，等价于对偶流形 $(\Phi,\nabla^2 H)$ 上的 Natural Gradient。

它不是说“任意坐标、任意近似算法的 Mirror Descent 都等于同一个 Natural Gradient”。

对 KL-MD，势函数取负熵时，primal 坐标是分布 $\pi$，对偶坐标是 logits / 指数族自然参数，对偶度量 $\nabla^2 H$ 正是 categorical / softmax 的 Fisher。于是：

- tabular 每状态精确 KL-MD $\iff$ 该状态分布单纯形上的精确 proximal 步；
- 对无穷小步长，它与该单纯形 Fisher--Rao 度量下的 NGD 一致；
- 有限步长 MD 是 proximal / 镜像步，不是线性化的 $F^{-1}g$ 一步。

神经网络参数 $\theta$ 一般**不是**这个对偶坐标。把 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta_k})$ 看成 $\theta$ 上的 Bregman 发散，通常不成立，因为 $\theta\mapsto\pi_\theta$ 不是指数族的自然参数映射。Raskutti--Mukherjee 因此不能直接套到 MDPO 的 $\theta$ 更新上。

## 局部二次化：唯一真正碰上 NGD 的地方

把 MDPO 目标在 $\theta_k$ 处展开。KL 项满足

$$
\nabla_\theta D_{\mathrm{KL}}\big|_{\theta_k}=0,
\qquad
\nabla_\theta^2
\mathbb{E}_s
D_{\mathrm{KL}}\bigl(\pi_\theta\Vert\pi_{\theta_k}\bigr)\big|_{\theta_k}
=
F(\theta_k).
$$

优势项线性化后，

$$
\Psi(\theta,\theta_k)
\approx
\mathrm{const}
+
g^{\top}(\theta-\theta_k)
-
\frac{1}{2t_k}(\theta-\theta_k)^{\top} F(\theta_k)(\theta-\theta_k),
$$

其中 $g=\nabla_\theta J(\theta_k)$ 就是普通政策梯度。这个局部二次问题的精确解是

$$
\theta-\theta_k
=
t_k F(\theta_k)^{-1}g.
$$

这正是一步 Natural Policy Gradient，步长为 $t_k$。TRPO 做的就是这个方向，再用 line search 卡 KL。

所以更准确的关系是：

$$
\boxed{
\text{MDPO 局部二次问题的精确 Newton 解}
=
\text{一步参数空间 NGD}
}
$$

$$
\boxed{
\text{MDPO 实际算法}
\neq
\text{NGD}
}
$$

中间隔了三层近似：

1. 用参数化策略类 $\Theta$ 代替完整单纯形上的精确 MD；
2. 用有限 $m$ 步欧氏 SGD 代替精确 $\arg\max\Psi$；
3. 真实的 $\Psi$ 并不是 $\theta$ 上的二次型，$F^{-1}g$ 只是它在 $\theta_k$ 处的局部 Newton 步。

若 inner loop 真用自然梯度去解 $\Psi$，并且步长、阻尼合适，它会更接近这个局部 Newton 点，也更接近“用 MD 近似 NGD”这句话。MDPO 论文刻意不用这条路。

## 和 TRPO 的对照

TRPO 每轮大致做：

$$
\theta_{k+1}
=
\theta_k
+
\eta F(\theta_k)^{-1}g,
$$

再对 $D_{\mathrm{KL}}(\pi_{\theta_k}\Vert\pi_\theta)\le\delta$ 做 line search。这是**显式 NGD**，外加硬约束。

On-policy MDPO 每轮做：对 $\Psi$ 走 $m$ 步欧氏 SGD，KL 以 penalty 出现，方向是 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta_k})$，不用 $F^{-1}$。论文把这当成优点：不必估计或求逆 Fisher。

因此：

- TRPO：一步（近似）NGD + 硬 KL 约束；
- MDPO：多步欧氏 SGD + 软 KL penalty，目标来自 MD；
- 二者都是 trust-region 政策优化，但几何不是同一个。

PPO 的 clip 更不是 NGD，也不是 MD 的精确 proximal 步。

## 对 OPD / F.5 局部几何的含义

F.5 的局部 OPD 目标已经是

$$
\min_{\Delta\theta}
\frac12\Delta\theta^{\top} A\Delta\theta-b^{\top}\Delta\theta,
\qquad
A=\mathbb{E}_c[J_c^{\top} F_c J_c].
$$

这里的 $A$ 既是 Gauss--Newton Hessian，也是 $\theta_0$ 处的 pullback Fisher。所以：

- 欧氏 GD：$\Delta\theta_s=[I-(I-\eta A)^s]A^{-1}b$；
- 自然梯度：$\Delta\theta_s^{\mathrm{NG}}=[1-(1-\eta)^s]A^{-1}b$；
- 若把 OPD 写成对教师的 KL proximal / Mirror Descent，并在参数空间把该 proximal 二次化后精确求解，得到的正是自然梯度 / Newton 步。

如果“用 MDPO 那套实现”去跑 OPD，即对

$$
\mathbb{E}_c
\bigl[
D_{\mathrm{KL}}\bigl(p_\theta(\cdot\mid c)\Vert p^\star(\cdot\mid c)\bigr)
\bigr]
$$

做多步欧氏 SGD，那么：

- 它是在最小化 OPD 目标，几何上接近 MD 的 penalty 形式；
- 它仍然不是 $A^{-1}(A\Delta\theta-b)$；
- 早期方向会更像欧氏 GD：先沿 $b$，再随谱旋转；
- 只有 inner loop 换成 Gauss--Newton / NGD，或者把局部二次问题解得足够准，才会得到 F.5 里那条方向从第一步就锁死的 NGD 轨迹。

这和 [mirror descent.md](/Users/fanghaotian/Desktop/obsidian/opd%20MD-NGD/mirror%20descent.md) 里的结论一致：用 MD 去近似 NGD 时，inner loop 不必解到机器精度，但停止标准应是自然梯度几何下的相对残差，而不是欧氏 loss 看起来不再下降。MDPO 默认并不按这个标准走。

## 一张对照表

| 对象                  | 在什么空间更新                                | 是否用 Fisher 逆                                 | 与 NGD 的关系                                              |
| --------------------- | --------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| 单纯形精确 KL-MD      | 分布 $\pi(\cdot\mid s)$                       | 不显式求逆，closed form 已含 softmax Fisher 几何 | 对偶坐标上精确等价于 NGD；有限步是 proximal 而非线性化 NGD |
| Raskutti--Mukherjee   | primal $\theta$ / dual $\mu=\nabla G(\theta)$ | 对偶度量 $\nabla^2 H$                            | 精确 MD $=$ 对偶流形 NGD                                   |
| TRPO / NPG            | 网络参数 $\theta$                             | 是，$F(\theta)^{-1}g$                            | 参数空间 NGD                                               |
| MDPO 局部二次的精确解 | 网络参数 $\theta$                             | 是，$t_k F^{-1}g$                                | 等于一步 NGD                                               |
| **MDPO 实际实现**     | 网络参数 $\theta$                             | **否，欧氏 SGD**                                 | **不等于 NGD**；$m=1$ 时退回 vanilla PG                    |

## 一句话

MDPO 的“Mirror Descent”是目标函数的来源，不是更新几何的实现。它用欧氏 SGD 近似 KL proximal 子问题；Natural Gradient 则是用 Fisher 度量直接走最速下降。二者只在“把该子问题在 $\theta_k$ 处二次化并精确求解”这一理想化步骤上重合。
