# F.5 OPD 动力学的局部几何视角

来源：Cai et al., _Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation_, [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)，附录 F.5（原文标题：A Local Geometric View of OPD Dynamics）。

本节给出一种局部几何分析，用来解释为何 On-Policy Distillation（OPD）会自然诱导低秩、并且早期锁定的更新方向，以及这种动力学如何不同于强化学习（RL）。通过对学生模型在基础模型附近做线性化，可以看到 OPD 目标的结构如何导出正文中观察到的经验现象。

## 设定与线性化

记一个 token 上下文为 $c=(x,y_{<t})$，其中 $x$ 是输入提示，$y_{<t}$ 是已经生成的前缀。定义：

- $z_{\theta}(c)\in\mathbb{R}^{V}$：参数为 $\theta$ 的学生模型 logits，词表大小为 $V$；
- $z^{\star}(c)\in\mathbb{R}^{V}$：固定教师模型的 logits；
- $\theta_0$：基础模型参数，也是 RL 与 OPD 训练的共同初始化；
- $\Delta\theta=\theta-\theta_0$：参数位移。

将 $z_{\theta}(c)$ 在 $\theta_0$ 附近做一阶展开：

$$
z_{\theta}(c)
=
z_{\theta_0}(c)
+
\underbrace{\frac{\partial z_{\theta}(c)}{\partial \theta}\bigg|_{\theta=\theta_0}}_{=:J_c}
\Delta\theta
+
O(\Vert\Delta\theta\Vert^2).
$$

其中 $J_c\in\mathbb{R}^{V\times\dim(\theta)}$ 是 logits 对参数的 Jacobian。在步长足够小、训练足够早的阶段，$\Vert\Delta\theta\Vert$ 很小，高阶项可以忽略：

$$
z_{\theta}(c)\approx z_{0}(c)+J_c\Delta\theta,
\qquad
z_{0}(c):=z_{\theta_0}(c).
$$

定义基础模型处的教师–学生 logit 残差：

$$
r_c:=z^{\star}(c)-z_{0}(c).
$$

于是 logit 差分为：

$$
z_{\theta}(c)-z^{\star}(c)\approx J_c\Delta\theta-r_c.
$$

## OPD 目标的局部二次近似

OPD 目标是在 on-policy 样本上最小化学生与教师之间的反向 KL：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_{\theta}(\cdot\mid x)}
\left[
D_{\mathrm{KL}}
\bigl(
\pi_{\theta}(\cdot\mid x,y_{<t})
\Vert
\pi^{\star}(\cdot\mid x,y_{<t})
\bigr)
\right].
$$

对固定上下文 $c$，记

$$
p_{\theta}(\cdot\mid c)=\mathrm{softmax}(z_{\theta}(c)),
\qquad
p^{\star}(\cdot\mid c)=\mathrm{softmax}(z^{\star}(c)).
$$

当两个分布接近时，KL 在 logit 空间中有二阶 Taylor 展开。令 $f(z)=D_{\mathrm{KL}}(p_z\Vert p^{\star})$，其中 $p_z=\mathrm{softmax}(z)$。则

$$
f(z)
\approx
f(z^{\star})
+
\underbrace{\nabla f(z^{\star})^{\top}(z-z^{\star})}_{=0}
+
\frac{1}{2}(z-z^{\star})^{\top}\nabla^2 f(z^{\star})(z-z^{\star}),
$$

因为 $z=z^{\star}$ 是最小值点，一阶导数为零。反向 KL 在教师点处的 Hessian 就是学生分布的 Fisher 信息矩阵：

$$
\nabla^2 f(z^{\star})
=
\mathrm{Diag}(p^{\star})-p^{\star}{p^{\star}}^{\top}
=:F_c^{\star}.
$$

因此，当 $z$ 靠近 $z^{\star}$ 时：

$$
D_{\mathrm{KL}}(p_z\Vert p^{\star})
\approx
\frac{1}{2}(z-z^{\star})^{\top} F_c^{\star}(z-z^{\star}).
$$

但局部分析是在 $\theta_0$ 附近线性化，因此学生 logits $z_{\theta}(c)$ 接近 $z_0(c)$，不一定接近 $z^{\star}(c)$。为了得到关于 $\Delta\theta$ 的二次型，可以把 Fisher 矩阵放在一个方便的分布上求值，通常取基础模型分布 $p_0(c)=\mathrm{softmax}(z_0(c))$。当 $z_{\theta}\approx z_0$ 且教师离基础模型不太远时，这个近似是自洽的。定义

$$
F_c:=\mathrm{Diag}(p_0(c))-p_0(c)p_0(c)^{\top}.
$$

于是近似

$$
D_{\mathrm{KL}}(p_{\theta}\Vert p^{\star})
\approx
\frac{1}{2}(z_{\theta}(c)-z^{\star}(c))^{\top} F_c(z_{\theta}(c)-z^{\star}(c))
=
\frac{1}{2}(J_c\Delta\theta-r_c)^{\top} F_c(J_c\Delta\theta-r_c).
$$

如果教师与基础模型已经相当对齐（蒸馏中很常见），则 $z^{\star}\approx z_0$ 且 $F_c\approx F_c^{\star}$。即使并非如此，这个二次型仍然给出 KL 的局部近似，最多相差一个与 $\Delta\theta$ 无关的加性常数，因为

$$
D_{\mathrm{KL}}(p_{\theta}\Vert p^{\star})
=
D_{\mathrm{KL}}(p_0\Vert p^{\star})
+
\nabla_{\theta}D_{\mathrm{KL}}(p_{\theta}\Vert p^{\star})\big|_{\theta_0}\Delta\theta
+
\frac{1}{2}\Delta\theta^{\top} H\Delta\theta
+\cdots,
$$

而 $\theta_0$ 处的 Hessian 涉及 $J_c^{\top} F_c^{\star} J_c$。把 $F_c$ 放在 $p_0$ 上求值，是神经正切核文献中的标准简化；当 $\Vert z^{\star}-z_0\Vert$ 较小时，它保留正确的二阶结构。

## 局部期望目标与梯度

对 on-policy 上下文 $c$ 取期望；到一阶，这些上下文可以由基础模型分布近似。于是

$$
\mathcal{L}_{\mathrm{OPD}}(\Delta\theta)
\approx
\frac{1}{2}
\mathbb{E}_{c}
\bigl[
(J_c\Delta\theta-r_c)^{\top} F_c(J_c\Delta\theta-r_c)
\bigr].
$$

展开二次型：

$$
\mathcal{L}_{\mathrm{OPD}}(\Delta\theta)
\approx
\frac{1}{2}\Delta\theta^{\top}
\underbrace{\mathbb{E}_c[J_c^{\top} F_c J_c]}_{=:A}
\Delta\theta
-
\Delta\theta^{\top}
\underbrace{\mathbb{E}_c[J_c^{\top} F_c r_c]}_{=:b}
+
\frac{1}{2}\mathbb{E}_c[r_c^{\top} F_c r_c].
$$

最后一项对 $\Delta\theta$ 是常数。因此局部目标是一个凸二次型：

$$
\mathcal{L}_{\mathrm{OPD}}(\Delta\theta)
=
\frac{1}{2}\Delta\theta^{\top} A\Delta\theta
-
b^{\top}\Delta\theta
+
\mathrm{const}.
$$

关于 $\Delta\theta$ 的梯度为

$$
g(\Delta\theta)
:=
\nabla_{\Delta\theta}\mathcal{L}_{\mathrm{OPD}}
=
A\Delta\theta-b.
$$

## 梯度下降动力学与闭式解

考虑步长固定为 $\eta>0$ 的梯度下降：

$$
\Delta\theta_{s+1}
=
\Delta\theta_s-\eta g(\Delta\theta_s)
=
\Delta\theta_s-\eta(A\Delta\theta_s-b)
=
(I-\eta A)\Delta\theta_s+\eta b.
$$

从 $\Delta\theta_0=0$（初始化在基础模型）出发，展开递推：

$$
\begin{aligned}
\Delta\theta_1
&=
\eta b, \\
\Delta\theta_2
&=
(I-\eta A)\eta b+\eta b
=
\eta\bigl[I+(I-\eta A)\bigr]b, \\
\Delta\theta_s
&=
\eta\sum_{j=0}^{s-1}(I-\eta A)^{j}b.
\end{aligned}
$$

这是矩阵几何级数。$A$ 是对称半正定的，因为它是 $J_c^{\top}F_c^{1/2}$ 的 Gram 矩阵。选取 $0<\eta<2/\lambda_{\max}(A)$ 以保证收敛。此时 $I-\eta A$ 的谱半径小于 $1$，级数收敛到

$$
\Delta\theta_{\infty}
=
\eta\bigl(I-(I-\eta A)\bigr)^{-1}b
=
A^{-1}b,
$$

其中 $A^{-1}$ 表示 $A$ 支撑上的伪逆。有限步公式有闭式：

$$
\Delta\theta_s
=
\bigl[I-(I-\eta A)^s\bigr]A^{-1}b.
$$

验证方式是因式分解：

$$
\sum_{j=0}^{s-1}(I-\eta A)^j
=
\bigl(I-(I-\eta A)^s\bigr)\bigl(I-(I-\eta A)\bigr)^{-1}
=
\bigl(I-(I-\eta A)^s\bigr)(\eta A)^{-1}.
$$

再乘以 $\eta b$ 即得。

## 自然梯度动力学与闭式解

上一节的梯度下降是欧氏空间中的最速下降。但在 F.5 的局部线性化里，$A$ 本身已经是学生模型在基础模型处的 pullback Fisher，也就是 Gauss--Newton 矩阵。把自然梯度接到同一套二次型上，可以得到更干净的方向动力学。

学生分布 $p_{\theta}(\cdot\mid c)=\mathrm{softmax}(z_{\theta}(c))$ 的 score 为

$$
\nabla_{\theta}\log p_{\theta}(y\mid c)
=
J_c^{\top}\bigl(e_y-p_{\theta}(\cdot\mid c)\bigr).
$$

在 $\theta_0$ 处冻结 Jacobian 与输出 Fisher $F_c$，则参数空间 Fisher 恰为

$$
F(\theta_0)
=
\mathbb{E}_c\bigl[J_c^{\top} F_c J_c\bigr]
=
A.
$$

因此，局部 OPD 损失的 Gauss--Newton Hessian 与 Fisher 度量重合。自然梯度更新是

$$
\Delta\theta_{s+1}
=
\Delta\theta_s
-
\eta A^{-1} g(\Delta\theta_s)
=
\Delta\theta_s
-
\eta A^{-1}(A\Delta\theta_s-b),
$$

其中 $A^{-1}$ 仍表示 $A$ 支撑上的伪逆。令 $P_A:=A^{-1}A$ 为到 $\mathrm{range}(A)$ 的正交投影，则

$$
\Delta\theta_{s+1}
=
(I-\eta P_A)\Delta\theta_s
+
\eta A^{-1}b.
$$

在 $A$ 的支撑上，$P_A$ 就是恒等，于是

$$
\Delta\theta_{s+1}
=
(1-\eta)\Delta\theta_s
+
\eta A^{-1}b.
$$

从 $\Delta\theta_0=0$ 出发：

$$
\begin{aligned}
\Delta\theta_1
&=
\eta A^{-1}b, \\
\Delta\theta_s
&=
\bigl[1-(1-\eta)^s\bigr] A^{-1}b.
\end{aligned}
$$

选取 $0<\eta<2$ 以保证 $\lvert 1-\eta\rvert<1$，则

$$
\Delta\theta_{\infty}^{\mathrm{NG}}
=
A^{-1}b,
$$

与欧氏梯度下降的极限相同。特别地，当 $\eta=1$ 时，自然梯度就是该二次型上的一步 Newton 法：$\Delta\theta_1=A^{-1}b$。

几何上，自然梯度是 Fisher--Rao / 输出 KL 几何中的最速下降。参数步长的度量是

$$
\Vert v\Vert_A^2
=
v^{\top} A v
=
\mathbb{E}_c\bigl[(J_c v)^{\top} F_c (J_c v)\bigr],
$$

也就是：一步参数更新有多大，要看它把 logits 在输出 Fisher 度量下推动了多远。欧氏梯度下降用 $\Vert v\Vert_2$ 量这一步，所以高曲率方向的欧氏梯度分量会被放大；自然梯度先用 $A^{-1}$ 把曲率除掉，再迈步。

与欧氏梯度下降的关键差别不在终点，而在路径。欧氏更新的第一步沿着 $b$，随后方向会随不同特征值的饱和速度而旋转；冻结度量下的自然梯度每一步都平行于 $A^{-1}b$，因此更新方向从第一步起就已经锁死，后续只放大幅度。这是比欧氏梯度下降更强的早期方向锁定。

若使用阻尼度量 $(A+\lambda I)^{-1}$，支撑上的极限仍是 $A^{-1}b$，但小特征值方向会被减慢，从而在自然梯度与欧氏梯度之间插值。实践中 $A$ 高度退化且病态，阻尼几乎是必要的。

### 相对欧氏梯度下降的好处

在这套局部二次型上，欧氏 GD 与自然梯度的极限相同，都是 $A^{-1}b$。自然梯度多出来的不是另一个解，而是一条更匹配目标几何的路径。

第一，曲率预条件。欧氏 GD 的稳定步长必须满足 $\eta<2/\lambda_{\max}(A)$，因此被最硬的方向卡住：大特征值方向很快饱和，小特征值方向几乎不动。自然梯度在支撑上只要求 $0<\eta<2$，所有 $\lambda_i>0$ 的方向同步推进。对 LLM 这种 $A$ 高度病态的情形，这意味着不必为了稳住 $\lambda_{\max}$ 而把有效步长压到接近零。

第二，几何匹配。OPD 要减小的是输出分布上的反向 KL，而

$$
\Vert v\Vert_A^2
=
\mathbb{E}_c\bigl[(J_c v)^{\top} F_c (J_c v)\bigr]
$$

正是“参数步 $v$ 把 logits 在输出 Fisher 度量下推动了多远”。自然梯度是这个度量下的最速下降；欧氏 GD 是 $\Vert v\Vert_2$ 下的最速下降，会把预算花在欧氏梯度大、但不一定是 Newton 方向的分量上。

第三，方向从第一步锁定。冻结度量下

$$
\Delta\theta_s^{\mathrm{NG}}
=
\bigl[1-(1-\eta)^s\bigr]A^{-1}b
$$

始终平行于 $A^{-1}b$。欧氏 GD 的第一步沿着 $b$，随后因各特征值饱和快慢而旋转。若关心的是论文中的早期方向锁定 / foresight，自然梯度给出的是更强、也更干净的版本。$\eta=1$ 时它就是该二次型上的一步 Newton 法，直接到达局部解。

第四，一阶参数化不变性。自然梯度在重新参数化下，函数空间中的步伐（到一阶）不变；欧氏 GD 会随坐标系拉伸而改变轨迹。

这些好处有明确边界。自然梯度并不创造更低秩的 $b$，也不改变局部凸二次的终点。若 $b$ 在很小的 $\lambda_i$ 上有噪声，系数 $\beta_i/\lambda_i$ 会被放大，所以实践中几乎必须加阻尼 $(A+\lambda I)^{-1}$。计算 $A^{-1}$ 本身也昂贵，通常只能用 K-FAC、共轭梯度等近似。

## 谱分解与方向动力学

令 $A=U\Lambda U^{\top}$ 为特征分解，其中 $\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\dots,\lambda_d)$，且 $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_d\ge 0$。令 $b=U\beta$，其中 $\beta_i=\langle b,u_i\rangle$。由于伪逆满足 $A^{-1}=U\Lambda^{-1}U^{\top}$，有

$$
A^{-1}b
=
\sum_{i:\lambda_i>0}\frac{\beta_i}{\lambda_i}u_i.
$$

同时 $(I-\eta A)^s=U(I-\eta\Lambda)^s U^{\top}$。因此

$$
\Delta\theta_s
=
U\bigl[I-(I-\eta\Lambda)^s\bigr]\Lambda^{-1}\beta
=
\sum_{i:\lambda_i>0}
\frac{1-(1-\eta\lambda_i)^s}{\lambda_i}
\beta_i u_i.
$$

这个表达式揭示了方向动力学。对每个特征方向 $u_i$，其贡献从零开始，渐近趋于 $\beta_i/\lambda_i$。因子 $1-(1-\eta\lambda_i)^s$ 在曲率 $\lambda_i$ 更大时增长更快，意味着 logits 对参数变化更敏感的方向会更早饱和。因此，如果许多方向上的投影 $\beta_i$ 消失，有效更新就会在整个训练过程中被限制在一个低维子空间中。

对应的自然梯度谱形式是

$$
\Delta\theta_s^{\mathrm{NG}}
=
\sum_{i:\lambda_i>0}
\bigl[1-(1-\eta)^s\bigr]
\frac{\beta_i}{\lambda_i}
u_i
=
\bigl[1-(1-\eta)^s\bigr] A^{-1}b.
$$

所有 $\lambda_i>0$ 的方向共享同一个增长因子 $1-(1-\eta)^s$，不再依赖曲率 $\lambda_i$。对照起来：

$$
\Delta\theta_s^{\mathrm{GD}}
=
\sum_{i:\lambda_i>0}
\frac{1-(1-\eta\lambda_i)^s}{\lambda_i}
\beta_i u_i,
\qquad
\Delta\theta_s^{\mathrm{NG}}
=
\bigl[1-(1-\eta)^s\bigr]
A^{-1}b.
$$

欧氏梯度下降中，高曲率方向先饱和，低曲率方向后追上，所以 $\Delta\theta_s$ 的方向会随 $s$ 旋转，除非 $b$ 几乎落在单一特征空间中。自然梯度把曲率预条件掉以后，方向从一开始就指向 $A^{-1}b$。两种算法的有效更新仍然被限制在 $\{\,u_i:\beta_i\neq 0\,\}$ 张成的子空间中；若 $b$ 集中在前 $k$ 个特征方向上，二者都会给出低秩轨迹。差别在于：欧氏梯度下降的早期低秩锁定同时依赖 $b$ 的集中性与谱隙带来的饱和快慢；自然梯度的早期方向锁定主要依赖 $b$ 的集中性，谱隙不再决定相对速度。

## 早期低秩锁定的充分条件

定义 $A$ 的前 $k$ 维特征空间为

$$
U_k=\mathrm{span}\{u_1,\dots,u_k\},
$$

并令 $P_{U_k}$ 为到该子空间的正交投影。假设驱动项 $b$ 集中在 $U_k$ 上，只留下一个小残差：

$$
\Vert P_{U_k^{\perp}}b\Vert
\le
\epsilon\Vert b\Vert,
\qquad
\epsilon\ll 1.
$$

等价地，分解

$$
b=b_{\parallel}+b_{\perp},
\qquad
b_{\parallel}=P_{U_k}b,
\qquad
b_{\perp}=P_{U_k^{\perp}}b.
$$

利用闭式动力学，更新可以写成

$$
\Delta\theta_s
=
\bigl[I-(I-\eta A)^s\bigr]A^{-1}b_{\parallel}
+
\bigl[I-(I-\eta A)^s\bigr]A^{-1}b_{\perp}.
$$

第一项落在主导特征空间 $U_k$ 中，第二项是来自 $U_k^{\perp}$ 的尾部贡献。我们不去假设 $A^{-1}$ 在正交补上具有压缩范数的作用，而是用有限步动力学的谱响应来控制尾部项。具体地，

$$
\bigl\Vert
\bigl[I-(I-\eta A)^s\bigr]A^{-1}b_{\perp}
\bigr\Vert
\le
\rho_{\perp}(s)\Vert b_{\perp}\Vert,
$$

其中

$$
\rho_{\perp}(s)
=
\max_{i>k,\,\lambda_i>0}
\frac{\bigl\lvert 1-(1-\eta\lambda_i)^s\bigr\rvert}{\lambda_i}.
$$

结合集中性假设，得到

$$
\bigl\Vert
\Delta\theta_s
-
\bigl[I-(I-\eta A)^s\bigr]A^{-1}b_{\parallel}
\bigr\Vert
\le
\rho_{\perp}(s)\epsilon\Vert b\Vert.
$$

因此，当投影后的 $b$ 高度集中在前 $k$ 维特征空间中时，尾部贡献在有限训练视界内保持很小。若再存在明显谱间隙

$$
\lambda_k\gg\lambda_{k+1},
$$

则 $U_k$ 中的主导方向会比尾部方向更早被激活并饱和。这就给性质 2（早期低秩锁定，Early Low-Rank Lock-in）提供了一个几何解释：优化路径大体被限制在训练早期就已经识别出的低维子空间中，后续优化主要是在该子空间内增大幅度，而不是去探索本质全新的方向。

### 为什么实践中 $b$ 是低秩的？

回忆

$$
b=\mathbb{E}_c[J_c^{\top} F_c r_c].
$$

残差

$$
r_c=z^{\star}(c)-z_0(c)
$$

是教师与基础模型的 logit 差。在蒸馏中，教师通常只在相对较少的功能关键 token 位置上锐化概率，例如关键推理 token、中间推理步骤、答案 token 或格式 token（Xu et al., 2026, TIP）。因此 $r_c$ 在有效支撑上常常是稀疏或低维的。Fisher 矩阵 $F_c$ 再按输出分布的局部几何对这些残差方向重新加权。尽管 $J_c$ 本身可以是高秩的，复合映射

$$
J_c^{\top} F_c r_c
$$

会把这个集中的残差信号投影回参数空间。对上下文取平均后，驱动项 $b$ 倾向于集中在最能影响这些关键 token 预测的参数方向上。这与正文第 3 节中观察到的 $\Delta W$ 低秩结构一致。

## 模块级抑制（功能冗余规避）

把参数分解为 $M$ 个模块（例如 embedding、attention、MLP 层），写成

$$
\Delta\theta=(\Delta\theta_1,\Delta\theta_2,\dots,\Delta\theta_M),
\qquad
J_c=[J_{c,1},J_{c,2},\dots,J_{c,M}],
$$

其中 $J_{c,m}=\partial z_{\theta}(c)/\partial\theta_m\big|_{\theta_0}$。模块 $m$ 的驱动项是

$$
b_m=\mathbb{E}_c[J_{c,m}^{\top} F_c r_c].
$$

曲率矩阵 $A$ 具有分块结构：

$$
A=
\begin{pmatrix}
A_{11} & A_{12} & \cdots & A_{1M} \\
A_{21} & A_{22} & \cdots & A_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
A_{M1} & A_{M2} & \cdots & A_{MM}
\end{pmatrix},
\qquad
A_{mn}=\mathbb{E}_c[J_{c,m}^{\top} F_c J_{c,n}].
$$

在局部最优 $\Delta\theta^*=A^{-1}b$（或梯度下降极限）处，有

$$
\sum_{n=1}^{M} A_{mn}\Delta\theta_n^*=b_m.
$$

若跨模块耦合较弱（即 $m\neq n$ 时 $A_{mn}$ 相对 $A_{mm}$ 很小），且 $A_{mm}$ 在其支撑上可逆，则

$$
\Delta\theta_m^*\approx A_{mm}^{-1}b_m.
$$

因此，若 $b_m\approx 0$（模块 $m$ 与教师残差弱耦合），则 $\Delta\theta_m^*\approx 0$。这就给性质 1（功能冗余规避，Functional Redundancy Avoidance）提供了一种机制：不能帮助匹配教师残差的模块几乎不会被更新。经验上，embedding 层以及 Transformer 底部/顶部层的 $b_m$ 较小，因而更新被抑制。

## 与强化学习动力学的比较

长度为 $T$ 的轨迹上，标准策略梯度更新（例如 PPO）为

$$
g_{\mathrm{RL}}
=
\mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_{\theta}(\cdot\mid x)}
\left[
\sum_{t=1}^{T} A_t\nabla_{\theta}\log\pi_{\theta}(y_t\mid c_t)
\right],
$$

其中 $c_t=(x,y_{<t})$，$A_t$ 是优势估计。利用 logit 参数化：

$$
\nabla_{\theta}\log\pi_{\theta}(y_t\mid c_t)
=
J_{c_t}^{\top}\bigl(e_{y_t}-p_{\theta}(\cdot\mid c_t)\bigr).
$$

因此

$$
g_{\mathrm{RL}}
=
\mathbb{E}
\left[
\sum_{t=1}^{T}
A_t J_{c_t}^{\top}
\bigl(e_{y_t}-p_{\theta}(\cdot\mid c_t)\bigr)
\right].
$$

相比之下，OPD 梯度（局部近似）为

$$
g_{\mathrm{OPD}}
=
-\nabla_{\Delta\theta}\mathcal{L}_{\mathrm{OPD}}
=
b-A\Delta\theta.
$$

在初始化处（$\Delta\theta=0$），$g_{\mathrm{OPD}}(0)=b$，它是教师残差的（除采样外）确定性函数。RL 在初始化处的梯度是

$$
g_{\mathrm{RL}}(0)
=
\mathbb{E}
\left[
\sum_{t}
A_t J_{c_t}^{\top}
\bigl(e_{y_t}-p_0(c_t)\bigr)
\right].
$$

两种范式的差异可以概括为几点。OPD 通过残差 $r_c$（再经 $F_c$ 过滤）获得稠密的 token 级监督；RL 依赖标量奖励 $A_t$，在稀疏奖励设定下大多数 token 上通常为零，因而 RL 梯度估计更嘈杂。此外，RL 的信用分配困难，因为 $A_t$ 依赖于整条轨迹和未来奖励，方差很高。OPD 中每个 token 的残差提供了更稳定的学习信号。最后，方向结构也本质不同：OPD 的驱动项 $b$ 继承了 $r_c$ 的低秩集中性；RL 的驱动项涉及 $e_{y_t}-p_0(c_t)$，这是一个在词表空间上具有全支撑的随机向量，因此更新更分散。

可以用梯度协方差近似来说明这种差异。对 OPD，初始化处的单样本梯度是

$$
\hat{g}_{\mathrm{OPD}}=J_c^{\top} F_c r_c,
$$

协方差为 $\Sigma_{\mathrm{OPD}}=\mathrm{Cov}(\hat{g}_{\mathrm{OPD}})$。对 RL，在单 token 简化（或忽略时间依赖）下，单样本梯度是

$$
\hat{g}_{\mathrm{RL}}=A J_c^{\top}(e_y-p_0(c)).
$$

其协方差满足

$$
\mathrm{Tr}(\Sigma_{\mathrm{RL}})
\approx
\mathbb{E}[A^2]\cdot\mathbb{E}\bigl[\Vert J_c^{\top}(e_y-p_0)\Vert^2\bigr]
\ge
\sigma_A^2\cdot\mathbb{E}\bigl[\Vert J_c^{\top}(e_y-p_0)\Vert^2\bigr],
$$

其中 $\sigma_A^2=\mathrm{Var}(A)$。在稀疏奖励设定下，$\sigma_A^2$ 可以很大，因为大多数轨迹奖励为零，只有少数轨迹非零。对 OPD，残差 $r_c$ 在许多 token 上非零，相对方差更低。此外，当 $r_c$ 较小时，$\Vert J_c^{\top}(e_y-p_0)\Vert$ 通常比 $\Vert J_c^{\top} F_c r_c\Vert$ 更大，因为 $F_c$ 的特征值至多为 $1$。因此实践中可以期望 $\mathrm{Tr}(\Sigma_{\mathrm{RL}})>\mathrm{Tr}(\Sigma_{\mathrm{OPD}})$，这意味着 OPD 沿着更平滑、更低噪声的优化轨迹前进。

## 小结

在局部体制下，OPD 可以近似为一个可能退化的凸二次最小化问题：

$$
\min_{\Delta\theta}
\frac{1}{2}\Delta\theta^{\top} A\Delta\theta
-
b^{\top}\Delta\theta.
$$

相应的梯度下降动力学具有谱形式

$$
\Delta\theta_s
=
\sum_{i:\lambda_i>0}
\frac{1-(1-\eta\lambda_i)^s}{\lambda_i}
\beta_i u_i.
$$

该表达式表明，每个特征方向上的更新由残差投影 $\beta_i=\langle b,u_i\rangle$、局部曲率 $\lambda_i$，以及有限步增长因子 $1-(1-\eta\lambda_i)^s$ 共同决定。

若改用冻结 Fisher 度量 $A$ 上的自然梯度，则

$$
\Delta\theta_s^{\mathrm{NG}}
=
\bigl[1-(1-\eta)^s\bigr] A^{-1}b,
$$

极限仍是 $A^{-1}b$，但所有支撑方向同步推进，更新方向从第一步起就保持不变。

如果驱动项 $b$ 集中在低维子空间中，例如 $A$ 的前 $k$ 维特征空间，并且存在明显谱间隙，那么从训练早期开始，更新就近似被限制在这个子空间中。这为早期低秩锁定提供了局部解释。

在模块层面，如果某个模块与教师残差的耦合可以忽略，即 $b_m\approx 0$，并且跨模块耦合项不占主导，则该模块的更新预期会被抑制。这解释了功能冗余规避。

与 RL 相比，OPD 的梯度信号更稠密、方差更低、方向更集中，这有助于解释 OPD 中观察到的更集中、更高效的更新模式。
