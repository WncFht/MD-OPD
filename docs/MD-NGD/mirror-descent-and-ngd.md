# Mirror Descent 与 NGD

你的理解有一半正确，但关键区别是：

> Mirror Descent 中虽然固定了旧策略 $\mu$，但完整的策略改进目标并不是所有期望都在 $\mu$ 下定义。通常是状态分布固定为 $d_\mu$，而动作分布仍然是候选新策略 $\pi$。

## 1. 论文中的实际 Mirror Descent 目标

典型的 Policy Mirror Descent 更新是

$$
\pi_{k+1}
=
\arg\max_{\pi}
\mathbb E_{s\sim d_{\pi_k}}
\left[
\mathbb E_{a\sim\pi}
Q^{\pi_k}(s,a)
-
\frac{1}{\eta_k}
D_h\left(\pi(\cdot\mid s),\pi_k(\cdot\mid s)\right)
\right].
$$

KL-Mirror Descent 中，$D_h$ 是 KL：

$$
D_h\left(\pi,\pi_k\right)
=
\mathrm{KL}\left(\pi\Vert\pi_k\right).
$$

这里有三个不同对象：

- 状态分布：$s\sim d_{\pi_k}$；
- 改进目标中的动作：$a\sim\pi$；
- KL 的参考策略：$\pi_k=\mu$。

因此它不是完全意义上的

$$
(s,a)\sim\mu.
$$

更准确地说是

$$
s\sim d_\mu,
\qquad
a\sim\pi,
\qquad
\mathrm{KL}(\pi\Vert\mu).
$$

## 2. Lan 的 Policy Mirror Descent：为什么没有显式 Importance Sampling？

Lan 的 [Policy Mirror Descent for Reinforcement Learning](https://doi.org/10.1007/s10107-022-01816-5) 在 Equation (3.5) 中使用

$$
\pi_{k+1}(\cdot\mid s)
=
\arg\min_{p\in\Delta(\mathcal A)}
\left\{
\eta_k
\langle Q^{\pi_k}(s,\cdot),p(\cdot\mid s)\rangle
+
D_h\left(p,\pi_k\right)
\right\}.
$$

在 tabular 情况下，$Q^{\pi_k}(s,a)$ 对所有动作都可以显式计算，因此

$$
\langle Q^{\pi_k}(s,\cdot),p(\cdot\mid s)\rangle
=
\sum_a p(a\mid s)Q^{\pi_k}(s,a)
$$

直接求和，不需要从旧策略采样动作，也不需要 Importance Sampling。

Lan 的随机版本只是用轨迹估计 $Q^{\pi_k}(s,a)$，然后把估计到的 $Q$ 代入 proximal 更新。这里没有 action-level importance ratio，原因是：

> 它估计的是每个 $(s,a)$ 的 action-value，然后在策略更新时对所有动作求和。

这和“从旧策略采样动作，再估计新策略下的动作期望”是两种不同的实现。

## 3. MDPO 明确显示：参数化策略中需要 ratio

Tomar 等人的 [Mirror Descent Policy Optimization](https://arxiv.org/abs/2005.09814) 在 Equation (6) 中写的是

$$
\max_\theta
\mathbb E_{s\sim\rho_{\pi_k}}
\left[
\mathbb E_{a\sim\pi_\theta}
A^{\pi_k}(s,a)
-
\frac{1}{t_k}
\mathrm{KL}
\left(
\pi_\theta\Vert\pi_k
\right)
\right].
$$

注意动作是从当前候选策略 $\pi_\theta$ 取样，而不是从 $\pi_k$ 取样。

但是实际数据是由 $\pi_k$ 生成的。因此在内层第 $i$ 次更新时，他们的梯度估计包含

$$
\frac{
\pi_{\theta_k^{(i)}}(a\mid s)
}{
\pi_{\theta_k}(a\mid s)
}
\nabla_\theta
\log\pi_{\theta_k^{(i)}}(a\mid s)
A^{\pi_k}(s,a).
$$

这个 ratio 正是 Importance Sampling ratio。

论文还明确指出：

- 只做一次 SGD 时，KL 梯度在 $\theta_k$ 处为零；
- 因而一次更新退化为 vanilla policy gradient；
- 多次 SGD 是近似求解 Mirror Descent 子问题；
- on-policy MDPO 使用普通 SGD，而不是 Natural Gradient。

因此，固定 $\mu$ 并不会自动消除 Importance Sampling。只要你用 $\mu$ 的动作样本去估计当前候选策略 $\pi_\theta$ 下的期望，就需要 ratio，除非你能够直接计算动作求和，或者重新从 $\pi_\theta$ 采样。

## 4. Forward KL 的 IS 推导

令

$$
\mu=\pi_k.
$$

KL-proximal 目标可以写成

$$
\mathcal M_k(\theta)
=
\mathbb E_{s\sim d_\mu,a\sim\pi_\theta}
\left[
A^\mu(s,a)
-
\frac{1}{\eta}
\log
\frac{\pi_\theta(a\mid s)}
{\mu(a\mid s)}
\right].
$$

如果手上的样本来自

$$
s\sim d_\mu,
\qquad
a\sim\mu,
$$

定义

$$
\rho_\theta(s,a)
=
\frac{\pi_\theta(a\mid s)}
{\mu(a\mid s)}.
$$

则等价的无偏估计是

$$
\mathcal M_k(\theta)
=
\mathbb E_{s\sim d_\mu,a\sim\mu}
\left[
\rho_\theta(s,a)
\left(
A^\mu(s,a)
-
\frac{1}{\eta}\log\rho_\theta(s,a)
\right)
\right].
$$

所以 forward KL 的完整 KL-MD 目标，在旧策略样本上一般需要 $\rho_\theta$。

但在初始点 $\theta=\theta_k$ 时，

$$
\rho_{\theta_k}(s,a)=1.
$$

因此第一次更新不需要非平凡的 Importance Sampling；这正是它退化为普通 on-policy policy gradient 的原因。

## 5. 为什么普通 Policy Gradient 看起来没有 $\mu$？

普通 on-policy Policy Gradient 写成

$$
\nabla J(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right].
$$

它不是“没有采样分布”，而是目标策略和行为策略相同：

$$
\text{target policy}=\text{behavior policy}=\pi_\theta.
$$

所以 ratio 是

$$
\frac{\pi_\theta}{\pi_\theta}=1.
$$

当你用旧策略 $\mu$ 的数据估计当前策略 $\pi_\theta$ 的 Policy Gradient 时，才出现

$$
\frac{\pi_\theta(a\mid s)}
{\mu(a\mid s)}.
$$

因此更准确的说法是：

> Importance Sampling 不是因为 Policy Gradient 没有 $\mu$，而是因为采样策略和目标策略不同。

Mirror Descent 也一样。它显式引入了旧策略 $\mu$，但目标中的候选动作分布仍然可能是 $\pi_\theta$。

## 6. Mirror Learning 论文的直接说明

[Mirror Learning: A Unifying Framework of Policy Optimisation](https://arxiv.org/abs/2201.02373) 的 Appendix F 对 off-policy 情况说得非常直接：

如果 replay buffer 中的动作来自历史策略 $\pi_{\mathrm{hist}}$，但目标需要计算

$$
\mathbb E_{a\sim\bar\pi}
Q^{\pi_{\mathrm{old}}}(s,a),
$$

则使用估计量

$$
\frac{
\bar\pi(a\mid s)
}{
\pi_{\mathrm{hist}}(a\mid s)
}
Q^{\pi_{\mathrm{old}}}(s,a).
$$

论文还验证了

$$
\mathbb E_{a\sim\pi_{\mathrm{hist}}}
\left[
\frac{\bar\pi(a\mid s)}
{\pi_{\mathrm{hist}}(a\mid s)}
Q^{\pi_{\mathrm{old}}}(s,a)
\right]
=
\mathbb E_{a\sim\bar\pi}
Q^{\pi_{\mathrm{old}}}(s,a).
$$

这直接说明：

> Off-policy Mirror Descent 在 replay action 上估计候选策略期望时，需要 Importance Sampling。

## 7. 但有些 off-policy MD 算法不显式使用 IS

Tomar 等人的 off-policy MDPO 使用 replay buffer 主要采样状态和训练 $Q$ 函数，然后通过 reparameterization trick 从当前策略 $\pi_\theta$ 重新采样动作。

因此它的 actor 目标可以写成

$$
s\sim\mathcal D,
\qquad
a=f_\theta(\epsilon,s),
\qquad
\epsilon\sim\mathcal N(0,I),
$$

动作确实来自当前策略，而不是 replay 中的历史策略，所以 actor 目标不需要显式 action ratio。

但这并不意味着 off-policy MDPO 完全没有分布偏差：

- replay 中的状态分布不一定是 $d_{\pi_k}$；
- $Q^{\pi_k}$ 是通过 off-policy critic 近似的；
- critic 的误差会传递给 Mirror Descent 更新。

也就是说，它是通过“重新采样当前动作 + off-policy critic”规避显式 action IS，而不是从理论上消除了 distribution shift。

## 8. 允许任意 replay 分布的 PMD 理论

Alfano、Yuan 和 Rebeschini 的 [A Novel Framework for Policy Mirror Descent with General Parameterization and Linear Convergence](https://arxiv.org/abs/2301.13139) 很有代表性。

它的理想目标在 Equation (11) 中是

$$
\mathbb E_{s\sim d_{\pi_t}_\mu}
\left[
\langle Q_t(s,\cdot),\pi_\theta(\cdot\mid s)\rangle
-
D_h\left(\pi_\theta,\pi_t\right)
\right].
$$

但它允许用任意状态 - 动作分布 $v_t$ 来训练近似器 $f_{\theta}$，不要求 $v_t=d_{\pi_t}_\mu\pi_t$。

代价是引入 concentrability coefficient，例如

$$
\mathbb E_{(s,a)\sim v_t}
\left[
\left(
\frac{
d_{\pi_t}_\mu(s)\pi(a\mid s)
}{
v_t(s,a)
}
\right)^2
\right].
$$

这说明：

> 可以不在算法代码中显式写 Importance Sampling，但理论上必须通过覆盖条件、分布失配系数或函数逼近误差来承担这个问题。

论文 Appendix A.2 也明确说，虽然这种框架允许 replay buffer 和 arbitrary behavior distribution，但 offline distribution shift 仍然是一个需要进一步研究的问题。

## 9. 最终结论

你的说法需要改成下面这样：

### 正确的部分

固定 $\mu$ 后，Mirror Descent 通常把状态分布冻结在旧策略：

$$
s\sim d_\mu.
$$

因此，对于这个局部 surrogate，不需要对状态分布做从 $d_\pi$ 到 $d_\mu$ 的修正。这是 TRPO、MDPO、PMD 中常见的设计。

### 不正确的部分

动作分布通常仍然是候选策略：

$$
a\sim\pi_\theta.
$$

所以：

- 如果能枚举所有动作：不需要 action IS；
- 如果从当前 $\pi_\theta$ 重新采样：不需要 action IS；
- 如果使用旧策略 $\mu$ 的动作样本：需要
    $$
    \frac{\pi_\theta(a\mid s)}
    {\mu(a\mid s)};
    $$
- 如果只在 $\theta=\theta_k$ 处做一步更新：ratio 等于 $1$，但 KL 也没有一阶作用；
- 如果从更旧的 replay behavior $b$ 采样：还会出现 $\pi_\theta/b$，并且状态分布失配需要额外处理。

一句话总结：

$$
\boxed{
\text{Mirror Descent 的 reference policy } \mu
\neq
\text{整个目标的 sampling distribution}
}
$$

更准确的结构是：

$$
\boxed{
s\sim d_\mu,\qquad
a\sim\pi_\theta,\qquad
D_{\mathrm{KL}}(\pi_\theta\Vert\mu)
}
$$

因此，Mirror Descent 本身不要求 Importance Sampling；但当你用旧策略或 replay buffer 的动作样本去估计候选新策略的动作期望时，Importance Sampling 仍然需要。完整检索报告保存在：[paper-search report](/Users/fanghaotian/Desktop/obsidian/paper-search-reports/20260910-050717-088619-policy-mirror-descent-importance-sampling-mirror-descent-off-pol.md)。
