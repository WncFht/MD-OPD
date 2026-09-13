# Mirror Descent

继续调研后的结论是：

> 用 Mirror Descent 近似 Natural Gradient 时，inner loop 不需要把子问题解到机器精度。最合理的停止标准是“Natural-gradient 几何下的相对最优性残差”，而不是 inner loss 看起来不再下降。  
> 实际上，先从相对残差 `1%--5%` 开始通常合理；`10%` 可用于高噪声、快速探索，低于 `0.1%` 往往只有在全批量、低噪声场景才值得。

**1. 文献中的核心关系**

1. Amari, _Natural Gradient Works Efficiently in Learning_，1998  
   [Neural Computation](https://doi.org/10.1162/089976698300017746)

    Natural Gradient 是 Fisher 度规下的最速下降：

    $$
    \delta_{\mathrm{NG}}
    =
    -\alpha F(\theta)^{-1}g,
    \qquad
    g=\nabla L(\theta).
    $$

2. Raskutti & Mukherjee, _The Information Geometry of Mirror Descent_, 2015  
   [arXiv:1310.7780](https://arxiv.org/abs/1310.7780)

    令 $G$ 严格凸，$H=G^\ast$ 为其 Legendre 对偶，$\mu=\nabla G(\theta)$。Mirror Descent：

    $$
    \theta_{t+1}
    =
    \arg\min_\theta
    \left\{
    \langle g_t,\theta\rangle
    +
    \frac{1}{\alpha_t}D_G(\theta,\theta_t)
    \right\}
    $$

    的一阶条件是：

    $$
    \nabla G(\theta_{t+1})
    =
    \nabla G(\theta_t)-\alpha_t g_t.
    $$

    换到对偶坐标 $\mu$ 后：

    $$
    \mu_{t+1}
    =
    \mu_t
    -
    \alpha_t
    \bigl[\nabla^2H(\mu_t)\bigr]^{-1}
    \nabla_\mu L.
    $$

    因而，论文的精确结论是：

    > $G$-Bregman Mirror Descent 在 primal 坐标中，等价于 dual Riemannian manifold 上的 Natural Gradient。

    它不是说“任意坐标下的 Mirror Descent 都等于同一个 Natural Gradient”。

3. Martens, _New Insights and Perspectives on the Natural Gradient Method_, 2020  
   [JMLR/arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

    这篇文章强调了三个实际问题：
    - Fisher 通常只能近似；
    - 经验 Fisher 不一定是真正的 Fisher；
    - 有限步长下，单纯的 $-\alpha F^{-1}g$ 可能过大，因此需要 damping 或 trust region。

    所以 inner loop 的误差只是总误差的一部分，不能修复 Fisher 估计错误或 Mirror Map 与 Fisher 不匹配的问题。

4. Yang & Toh, _Bregman Proximal Point Algorithm Revisited_, 2022  
   [arXiv:2105.10370](https://arxiv.org/abs/2105.10370)

    这篇文章专门分析 Bregman 子问题被近似求解时，外层算法还能否收敛。

5. Guigues, _Inexact Stochastic Mirror Descent for Two-Stage Nonlinear Stochastic Programs_, 2018/2021  
   [arXiv:1805.11732](https://arxiv.org/abs/1805.11732)

    在随机 Mirror Descent 中，理论上通常要求 inner error $\varepsilon_t\to0$；实际实现可以固定 inner iteration 数，但这只是工程 heuristic，不存在跨问题通用的固定步数。

6. Dvurechensky et al., _Generalized Mirror Prox for Monotone Variational Inequalities: Universality and Inexact Oracle_, 2018  
   [arXiv:1806.05140](https://arxiv.org/abs/1806.05140)

    该文表明，prox 子问题的不可控误差会直接形成外层算法的误差下界，也就是 noise floor。

**2. Inner loop 实际在解什么**

设外层已经计算出 $g_t$，并冻结 $\theta_t$ 和 $g_t$。Inner loop 应该求解 Mirror-Prox 子问题：

$$
q_t(\theta)
=
\langle g_t,\theta-\theta_t\rangle
+
\frac{1}{\alpha_t}D_\psi(\theta,\theta_t).
$$

精确 Mirror Descent 解 $\theta_t^+$ 满足：

$$
0
=
\nabla q_t(\theta_t^+)
=
g_t
+
\frac{1}{\alpha_t}
\left[
\nabla\psi(\theta_t^+)-\nabla\psi(\theta_t)
\right].
$$

因此，给定 inner SGD 返回的 $\widehat\theta_t$，最自然的残差是：

$$
r_t
=
g_t
+
\frac{1}{\alpha_t}
\left[
\nabla\psi(\widehat\theta_t)-\nabla\psi(\theta_t)
\right].
$$

精确求解意味着 $r_t=0$。

注意：如果 inner loop 直接继续优化原始训练损失，而没有冻结 $g_t$ 和镜像中心 $\theta_t$，那它就不是在近似 Mirror Descent 的 prox 子问题。

**3. 为什么要用 Fisher 对偶范数衡量残差**

设

$$
F_t=\nabla^2\psi(\theta_t).
$$

它定义局部范数：

$$
\Vert v\Vert_{F_t}
=
\sqrt{v^\top F_t v},
$$

以及协向量的对偶范数：

$$
\Vert r\Vert_{F_t^{-1}}
=
\sqrt{r^\top F_t^{-1}r}.
$$

在局部二次情形：

$$
D_\psi(\theta_t+d,\theta_t)
\approx
\frac12d^\top F_t d.
$$

这时 Natural Gradient 子问题是：

$$
q_t(d)
=
g_t^\top d
+
\frac{1}{2\alpha_t}d^\top F_t d,
$$

其精确解为：

$$
d_t^\star
=
-\alpha_tF_t^{-1}g_t.
$$

若 inner loop 返回 $\widehat d_t$，定义二次模型残差：

$$
r_t
=
g_t+\frac{1}{\alpha_t}F_t\widehat d_t.
$$

则有精确关系：

$$
\widehat d_t-d_t^\star
=
\alpha_tF_t^{-1}r_t,
$$

并且：

$$
\frac{
\Vert\widehat d_t-d_t^\star\Vert_{F_t}
}{
\Vert d_t^\star\Vert_{F_t}
}
=
\frac{
\Vert r_t\Vert_{F_t^{-1}}
}{
\Vert g_t\Vert_{F_t^{-1}}
}.
$$

这给出了一个非常直接的结论：

> 在局部二次情形，Natural-gradient 几何下的相对残差，恰好就是更新方向的相对误差。

因此推荐的停止规则是：

$$
\frac{
\Vert r_t\Vert_{F_t^{-1}}
}{
\Vert g_t\Vert_{F_t^{-1}}+\varepsilon_{\mathrm{floor}}
}
\le
\tau_{\mathrm{in}}.
$$

如果只能计算普通欧氏范数，则有条件数损失：

$$
\frac{
\Vert\widehat d_t-d_t^\star\Vert_{F_t}
}{
\Vert d_t^\star\Vert_{F_t}
}
\lesssim
\sqrt{\kappa(F_t)}
\frac{\Vert r_t\Vert_2}{\Vert g_t\Vert_2}.
$$

所以 Fisher 条件数很大时，单纯要求 $\Vert r_t\Vert_2/\Vert g_t\Vert_2$ 很小仍然可能不够。

**4. 如果用 inner objective gap 作为停止标准**

设 $q_t$ 对某个范数是 $m_t$-强凸的，返回点满足：

$$
q_t(\widehat\theta_t)-q_t(\theta_t^+)\le\delta_t.
$$

则：

$$
\Vert\widehat\theta_t-\theta_t^+\Vert
\le
\sqrt{\frac{2\delta_t}{m_t}}.
$$

若希望更新的相对误差不超过 $\tau_{\mathrm{in}}$，充分条件是：

$$
\delta_t
\le
\frac{m_t}{2}
\tau_{\mathrm{in}}^2
\Vert\theta_t^+-\theta_t\Vert^2.
$$

在局部 Natural-gradient 二次模型中，这近似为：

$$
\delta_t
\lesssim
\frac{\alpha_t}{2}
\tau_{\mathrm{in}}^2
\Vert g_t\Vert_{F_t^{-1}}^2.
$$

因此，单纯监控 inner loss 的绝对下降量并不合理；阈值应随着外层步长和梯度大小变化。

**5. SGD inner loop 的噪声下限**

如果 inner SGD 的梯度噪声方差为 $\sigma_{\mathrm{in}}^2$，且 inner objective 的强凸、光滑参数分别为 $m_t,L_t$，固定 inner learning rate $a$ 时，典型结果是：

$$
\mathbb E[q_t(z_K)-q_t^\star]
\lesssim
(1-a m_t)^K\Delta_0
+
\frac{aL_t\sigma_{\mathrm{in}}^2}{m_t}.
$$

第一项是优化误差，第二项是随机噪声造成的误差地板。

这意味着：

- inner SGD 不能无限逼近精确 prox 点；
- 当优化误差已经低于噪声地板时，继续迭代基本没有意义；
- 如果 outer minibatch 本身的 Fisher-gradient 噪声已经达到 $5\%$，把 inner residual 压到 $0.01\%$ 通常没有收益。

采用递减 inner learning rate 时，常见的是：

$$
\mathbb E[q_t(z_K)-q_t^\star]
=
O\left(\frac{1}{K}\right),
$$

但所需的 $K$ 仍取决于 inner objective 的条件数和噪声方差，不存在统一固定步数。

**6. 文献中的严格误差条件**

Yang–Toh 的 inexact Bregman proximal 条件可以写成：

$$
\Delta_t
\in
\partial_{\nu_t}f(\widetilde\theta_t)
+
\gamma_t
\left[
\nabla\psi(\theta_{t+1})-\nabla\psi(\theta_t)
\right],
$$

并满足：

$$
\Vert\Delta_t\Vert\le\varepsilon_t^{\mathrm{res}},
\qquad
D_\psi(\widetilde\theta_t,\theta_{t+1})
\le
\mu_t^{\mathrm{geom}}.
$$

其中：

- $\varepsilon_t^{\mathrm{res}}$ 控制一阶最优性残差；
- $\mu_t^{\mathrm{geom}}$ 控制返回点和 prox 点之间的 Bregman 距离；
- $\nu_t$ 控制次梯度或函数值误差。

对于固定 $\gamma_t$ 的凸问题，为保持标准外层速率，一个保守的充分条件是这些误差可求和：

$$
\sum_t\mu_t^{\mathrm{geom}}<\infty,
\qquad
\sum_t\gamma_t^{-1}\varepsilon_t^{\mathrm{res}}<\infty,
\qquad
\sum_t\gamma_t^{-1}\nu_t<\infty.
$$

若还要控制论文中的加权下降误差，一个简单的充分 schedule 是：

$$
\mu_t^{\mathrm{geom}},
\varepsilon_t^{\mathrm{res}},
\nu_t
=
O(t^{-p}),
\qquad p>2.
$$

但这是确定性凸 BPPA 理论，不能直接当作深度学习随机非凸训练的必要条件。

**7. 实际建议**

如果你现在实现“Mirror Descent 近似 Natural Gradient”，我建议：

1. 外层计算并冻结 $g_t$、$\theta_t$、$F_t$ 或 $\nabla^2\psi(\theta_t)$。
2. Inner SGD 优化 $q_t$，而不是继续优化原始 loss。
3. 每隔若干步计算：

    $$
    r_t
    =
    g_t+
    \frac{1}{\alpha_t}
    \left[
    \nabla\psi(\widehat\theta_t)-\nabla\psi(\theta_t)
    \right].
    $$

4. 以如下指标停止：

    $$
    \rho_t
    =
    \frac{
    \Vert r_t\Vert_{F_t^{-1}}
    }{
    \Vert g_t\Vert_{F_t^{-1}}+\varepsilon_{\mathrm{floor}}
    }.
    $$

5. 工程上可以这样设：
    - $\rho_t\le0.1$：高噪声或快速探索；
    - $\rho_t\le0.05$：一般推荐起点；
    - $\rho_t\le0.01$：低噪声、全批量或对方向精度敏感；
    - $\rho_t\le10^{-3}$：通常只适合确定性实验，随机训练中常常浪费计算。

还要设置最大 inner iteration 数，但固定步数只能作为预算上限，不能作为唯一停止条件。

最后，必须区分三种误差：

$$
\text{总误差}
\approx
\underbrace{\text{Mirror Descent 与 Natural Gradient 的有限步误差}}_{O(\alpha_t)}
+
\underbrace{\text{inner solve 误差}}_{\rho_t}
+
\underbrace{\text{梯度/Fisher 估计误差}}_{\text{minibatch noise}}.
$$

当 $\psi$ 的 Hessian 与 Fisher 不匹配时，inner SGD 即使完全收敛，也只是精确求解了“错误的几何”。这点比 inner loop 是否多跑几十步更重要。

这也和你前面那篇谱球论文的统一视角一致：论文把更新范数及其对偶映射作为核心；Natural Gradient 是其中的光滑二次几何特例，而 Mirror Descent 是更一般的 Bregman 几何。该论文自身没有给出 inner solve 的误差理论，相关条件应参考上面的 inexact Bregman proximal / stochastic Mirror Descent 文献。[本地论文第 2 节](/Users/fanghaotian/Downloads/translate_zh_CN 4/section/02related.tex:47) / [理论附录](/Users/fanghaotian/Downloads/translate_zh_CN 4/section/appendix.tex:7)
