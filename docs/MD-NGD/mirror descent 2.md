# Mirror Descent 2

先给可执行结论：

如果每个 outer step 都重新求一个 Mirror Descent 的 Bregman/KL proximal 子问题，我建议把内层 SGD 停在

$$
\left\Vert r_t\right\Vert_{F_t^{-1}}
\le
\rho_t\eta_t
\left\Vert \widehat g_t\right\Vert_{F_t^{-1}},
\qquad
\rho_t\approx 0.1\text{--}0.3,
$$

其中 $\rho_t$ 是“内层误差占理想 Natural-Gradient 步长的比例”。$\rho<1$ 是很多凸的 inexact proximal 理论中的门槛；$0.1$--$0.3$ 是工程起点，不是普适定理。若内层噪声地板已经高于这个目标，继续跑 SGD 没有意义，应增加 batch、减小 inner learning rate 或使用方差降低。

## 1. Mirror Descent 为什么能近似 Natural Gradient

令外层梯度为 $\widehat g_t$，Mirror Descent 子问题为

$$
\theta_{t+1}^{\star}
=
\arg\min_{\theta}
\left\{
\left\langle \widehat g_t,\theta-\theta_t\right\rangle
+
\frac{1}{\eta_t}
D_{\psi}(\theta,\theta_t)
\right\}.
$$

一阶最优条件是

$$
\nabla\psi(\theta_{t+1}^{\star})
=
\nabla\psi(\theta_t)-\eta_t\widehat g_t.
$$

如果

$$
M_t=\nabla^2\psi(\theta_t),
$$

那么小步长下

$$
D_{\psi}(\theta_t+\delta,\theta_t)
=
\frac12\delta^\top M_t\delta
+
O(\Vert\delta\Vert^3).
$$

因此

$$
\theta_{t+1}^{\star}-\theta_t
\approx
-\eta_tM_t^{-1}\widehat g_t.
$$

当 $M_t=F_t$，其中 $F_t$ 是 Fisher 信息矩阵时，就得到局部 Natural Gradient：

$$
\delta_t^{\mathrm{NG}}
=
-\eta_tF_t^{-1}\widehat g_t.
$$

Raskutti 和 Mukherjee 的结果更精确：Bregman Mirror Descent 在原坐标中进行时，等价于对偶坐标中的 Natural Gradient，度规是 $\nabla^2\psi^*$。所以：

- $\psi$ 是二次函数且 Hessian 恒定时，MD 与固定 Mahalanobis Natural Gradient 完全一致；
- 在指数族或统计流形中，可以在对偶坐标中得到严格等价；
- 一般 Fisher 场不一定是某个全局势函数的 Hessian，因此通常只能说是局部或近似等价；
- 有限步长时，Fisher 只是 KL 的二阶近似，仍需要 damping 或 trust region。

例如概率单纯形上取负熵

$$
\psi(p)=\sum_i p_i\log p_i,
$$

其 Bregman 散度是 KL，Mirror Descent 更新为

$$
p_{t+1,i}
\propto
p_{t,i}\exp(-\eta_t g_{t,i}).
$$

这是有限步的 KL-Mirror Descent；Natural Gradient 是它在小步长下的 Fisher-Rao 局部形式。

## 2. 内层 SGD 应该监控什么

内层 SGD 不是在追求原始训练 loss 收敛，而是在近似上面的 proximal 子问题。

定义内层输出为 $\widetilde\theta_{t+1}$，定义 dual/Fisher 残差

$$
r_t
=
\nabla\psi(\widetilde\theta_{t+1})
-
\nabla\psi(\theta_t)
+
\eta_t\widehat g_t.
$$

精确求解时 $r_t=0$。

Fisher 度量和对应的 dual norm 为

$$
\Vert v\Vert_{F_t}
=
\sqrt{v^\top F_tv},
\qquad
\Vert u\Vert_{F_t^{-1}}
=
\sqrt{u^\top F_t^{-1}u}.
$$

推荐停止条件是

$$
\boxed{
\Vert r_t\Vert_{F_t^{-1}}
\le
\rho_t\eta_t
\Vert\widehat g_t\Vert_{F_t^{-1}}
}
$$

等价地，也可以写成

$$
\Vert\nabla\Phi_t(\widetilde\theta_{t+1})\Vert_{F_t^{-1}}
\le
\rho_t
\Vert\widehat g_t\Vert_{F_t^{-1}},
$$

其中 $\Phi_t$ 是内层 proximal 目标。

在二次镜像函数情形，

$$
\psi(\theta)=\frac12\theta^\top F_t\theta,
$$

有

$$
r_t
=
F_t(\widetilde\theta_{t+1}-\theta_{t+1}^{\star}),
$$

所以停止条件严格等价于

$$
\Vert\widetilde\theta_{t+1}-\theta_{t+1}^{\star}\Vert_{F_t}
\le
\rho_t
\Vert\theta_{t+1}^{\star}-\theta_t\Vert_{F_t}.
$$

也就是说，内层解的误差不超过理想 Natural-Gradient 步长的 $\rho_t$ 倍。

这比监控普通的 $\ell_2$ 梯度范数更合理，因为 Fisher 很可能病态，欧氏范数无法反映真正的函数分布变化。

## 3. 文献中的误差结论

| 文献                              | 主要结论                                                                     | 对内层 SGD 的启示                                |
| --------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------ |
| Amari, 1998                       | Natural Gradient 是 Fisher 度规下的最速下降                                  | 误差应该用 Fisher 几何衡量                       |
| Raskutti & Mukherjee, 2015        | Bregman MD 等价于对偶流形上的 NG                                             | 需要控制 dual mirror-map 残差                    |
| Martens, 2020                     | Fisher 是 KL 的局部二阶近似；有限步需要 damping/trust region                 | 内层求得再精确，几何近似错误仍然存在             |
| Pascanu & Bengio, 2014            | 用 CG 解 Fisher 线性系统，示例使用 `max iters=20`、`rtol=1e-4`               | 这是线性求解器实例，不是 SGD 的普适阈值          |
| Choi et al., 2023                 | inexact proximal MD 的动态 regret 中显式累积内层误差                         | 固定绝对误差会造成误差地板                       |
| Yang & Toh, 2022                  | Bregman proximal 方法允许相对误差 $\sigma<1$                                 | 相对 Bregman 误差比绝对 loss 阈值更自然          |
| Yang & Toh, 2021/2026             | 在相对光滑条件下通常要求 $\lambda>L$ 且 $\sigma<1-L/\lambda$                 | 若 proximal 正则较强，允许的内层误差更大         |
| Hong et al., 2020/2023            | 两时间尺度算法不要求每轮内层精确收敛，只需 $\alpha_t/\beta_t\to0$            | persistent inner loop 可以每轮只做少量 SGD       |
| Dembo, Eisenstat & Steihaug, 1982 | inexact Newton 的相对残差要求 $\eta_t<1$；若 $\eta_t\to0$ 可获得更强局部收敛 | 支持“相对残差小于 1”是理论门槛，但不是工程最佳值 |

Yang–Toh 的理论还说明：如果采用绝对误差，需要误差序列满足某种可求和条件，例如

$$
\sum_t\varepsilon_t<\infty
$$

或相应的加权可求和条件。否则通常只能收敛到一个误差邻域。

因此要区分：

- 理论型条件：相对误差小于 $1$，或绝对误差逐渐衰减且可求和；
- 工程型条件：通常从 $\rho=0.1$--$0.3$ 开始；
- 精确算法：令 $\rho_t\to0$，但在随机神经网络训练中通常不值得。

## 4. 为什么不能盲目把内层 SGD 跑到很小

最终误差可以分解为

$$
\widetilde\delta_t-\delta_t^{\mathrm{NG}}
=
\underbrace{\delta_t^{\mathrm{MD},\star}
-\delta_t^{\mathrm{NG}}}_{\text{镜像几何与 Fisher 的差异}}
+
\underbrace{\widetilde\delta_t
-\delta_t^{\mathrm{MD},\star}}_{\text{内层 SGD 误差}}
+
\underbrace{\text{梯度/Fisher 估计误差}}_{\text{随机噪声}}.
$$

即使内层 SGD 完全收敛，第一项和第三项仍然存在。

如果外层梯度噪声协方差为 $\Sigma_{g,t}$，则 Natural-Gradient 步的噪声规模大致为

$$
\mathbb E
\left[
\Vert
\eta_tF_t^{-1}\xi_t
\Vert_{F_t}^{2}
\right]
=
\eta_t^2
\operatorname{tr}
\left(F_t^{-1}\Sigma_{g,t}\right).
$$

因此内层误差不必显著小于

$$
\eta_t^2
\operatorname{tr}
\left(F_t^{-1}\Sigma_{g,t}\right).
$$

一个实用准则是

$$
\mathbb E\Vert e_t\Vert_{F_t}^{2}
\lesssim
c^2\eta_t^2
\operatorname{tr}
\left(F_t^{-1}\Sigma_{g,t}\right),
\qquad
c\in[0.1,0.3].
$$

如果 empirical Fisher、mini-batch gradient 或镜像势本身误差已经达到 $20\%$，把 inner SGD 误差继续压到 $0.01\%$ 基本是在浪费计算。

Martens 还特别指出，empirical Fisher 不一定等于真正的 Fisher；Pascanu 和 Bengio 也强调，Fisher 只在局部有效，且 minibatch 上做过于激进的 line search 可能导致过拟合当前 batch。

## 5. 如何把容差换算成 SGD 步数

假设内层 proximal 目标在 Fisher 范数下是 $\mu_t$-strongly convex、$L_t$-smooth，使用 inner learning rate $\gamma_{\mathrm{in}}$。

确定性 GD 有近似收缩

$$
\Delta_{K,t}
\lesssim
(1-\gamma_{\mathrm{in}}\mu_t)^K
\Delta_{0,t},
$$

其中 $\Delta_{K,t}$ 是 inner objective gap。

为了达到相对步长误差 $\rho_t$，一个充分的目标 gap 是

$$
\Delta_{\mathrm{target},t}
\le
\frac{\mu_t}{2}
\rho_t^2
\eta_t^2
\Vert\widehat g_t\Vert_{F_t^{-1}}^2.
$$

所以所需迭代数大致为

$$
K_t
\gtrsim
\frac{1}{\gamma_{\mathrm{in}}\mu_t}
\log
\frac{\Delta_{0,t}}
{\Delta_{\mathrm{target},t}}.
$$

取 $\gamma_{\mathrm{in}}\approx1/L_t$ 时，

$$
K_t
=
O\left(
\frac{L_t}{\mu_t}
\log
\frac{\Delta_{0,t}}
{\Delta_{\mathrm{target},t}}
\right).
$$

对于随机 SGD，典型结构是

$$
\mathbb E
\Vert\theta_K-\theta^\star\Vert_{F_t}^{2}
\lesssim
(1-\gamma_{\mathrm{in}}\mu_t)^K
\Vert\theta_0-\theta^\star\Vert_{F_t}^{2}
+
C\frac{\gamma_{\mathrm{in}}\sigma_{\mathrm{in}}^{2}}{\mu_t}.
$$

第二项是噪声地板。如果它已经大于目标误差，再增加 $K$ 也不会继续改善。

神经网络参数化下，内层目标通常不是强凸的，此时不能使用上述全局迭代数保证。应改为：

- Fisher/Bregman-preconditioned gradient norm；
- proximal KKT residual；
- primal-dual gap；
- achieved KL 是否在 trust-region 范围内；
- 新 batch 上的 surrogate improvement 是否仍然成立。

## 6. 实际建议

如果是双循环算法，每个 outer step 重新求 KL/Bregman prox：

1. 先用 $\rho=0.2$。
2. 如果训练不稳定或 outer step 对误差敏感，收紧到 $\rho=0.1$。
3. 如果 batch 很小、inner noise 很大，可以放宽到 $\rho=0.3$。
4. $\rho=0.5$ 仍可能工作，但更像粗略的 proximal step。
5. $\rho\approx1$ 只是理论上“没有发散”的边界，通常不足以称为较好的 Natural-Gradient 近似。
6. 如果 residual 卡住，优先增加 batch 或减小 inner learning rate，而不是无限增加 inner steps。
7. residual 最好用 fresh mini-batch 或滑动平均估计，避免在同一 batch 上过拟合。

如果采用 persistent inner loop，则不需要每个 outer step 求到很高精度。可以让 inner step size $\beta_t$ 比 outer step size $\alpha_t$ 更快，满足

$$
\frac{\alpha_t}{\beta_t}\to0.
$$

Hong 等人的强凸例子使用了类似

$$
\alpha_t=O(t^{-1}),
\qquad
\beta_t=O(t^{-2/3}),
$$

的两时间尺度设计。这里的关键不是每轮“完全收敛”，而是内层跟踪误差要比 outer 参数变化更快地消失。

另外，如果是 tabular policy 或可以直接计算动作求和的模型，KL-Mirror Descent 往往有闭式更新，根本不需要 inner SGD。只有在神经策略参数化、受限策略族或需要投影回参数空间时，才需要用 SGD 近似这个 prox。

## 调研报告与来源诊断

本次 `paper-search` 使用了三个查询，时间范围为 1990--2026，共得到 119 篇去重结果，主要来自 arXiv、OpenReview、OpenAlex 和 Crossref。完整报告在：

- [完整检索报告](/Users/fanghaotian/Desktop/obsidian/paper-search-reports/20260911-073543-558282-inexact-mirror-descent-bregman-proximal-relative-error-inner-sol.md)
- [Natural Gradient 与 Mirror Descent 等价性](/Users/fanghaotian/Desktop/obsidian/paper-search-reports/20260911-072052-370768-natural-gradient-mirror-descent-equivalence-fisher-bregman-natur.md)
- [Inexact Natural Gradient / inner loop](/Users/fanghaotian/Desktop/obsidian/paper-search-reports/20260911-072052-378106-inexact-natural-gradient-inner-loop-approximate-natural-gradient.md)
- [两时间尺度随机近似](/Users/fanghaotian/Desktop/obsidian/paper-search-reports/20260911-072944-530163-two-timescale-stochastic-approximation-inner-loop-outer-loop-con.md)

检索诊断也保留在报告中：Semantic Scholar 返回 HTTP 429，DBLP 返回 `JSONDecodeError`，OpenAlex 有两条空标题记录被丢弃。因此上面的结论主要依赖已经核对过的经典论文和 arXiv/OpenReview/Crossref 结果。
