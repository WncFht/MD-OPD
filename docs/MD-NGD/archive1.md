# Archive 1

不是严格的 MD-OPD，也不严格等于“$\mu$ 走一步步长 $\eta$ 的 NGD”。

令 rollout 策略为 $\mu=\pi_{\theta_k}$，固定 rollout 上下文分布为 $d_\mu$。用旧策略样本、通过 Importance Sampling 估计当前策略的 OPD 时，目标通常是

$$
P_k(\theta)
=
\mathbb E_{c\sim d_\mu,\;a\sim\mu}
\left[
\frac{\pi_\theta(a\mid c)}{\mu(a\mid c)}
\log\frac{\pi_\theta(a\mid c)}{q(a\mid c)}
\right]
+
\frac{1}{\eta}
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}(\pi_\theta\Vert\mu).
$$

这里 IS 只是在估计

$$
L_k(\theta)
=
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}(\pi_\theta\Vert q);
$$

它不会把该目标变成 mirror descent。

你保留了完整的 $L_k(\theta)$，所以多轮 PPO epoch 实际上是在近似求解

$$
\min_\theta
L_k(\theta)
+
\frac{1}{\eta}
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}(\pi_\theta\Vert\mu).
$$

这应叫做 KL-proximal OPD，而非严格 MD-OPD。严格的 MD-OPD 应把完整 OPD 损失替换成其在 $\mu$ 处的一阶线性化：

$$
\min_\theta
g_k^\top(\theta-\theta_k)
+
\frac{1}{\eta}
\mathbb E_{c\sim d_\mu}
D_{\mathrm{KL}}(\pi_\theta\Vert\mu),
\qquad
g_k=\nabla_\theta L_k(\theta_k).
$$

局部地，令 $F_k$ 是 $\mu$ 处 Fisher，$H_k=\nabla^2 L_k(\theta_k)$。若 epoch 足够多，确实近似解到了 proximal 子问题的最优点，则

$$
\theta_{k+1}-\theta_k
\approx
-
\left(H_k+\frac1\eta F_k\right)^\dagger g_k.
$$

而一步步长为 $\eta$ 的 NGD 是

$$
\theta_{k+1}-\theta_k
=
-\eta F_k^\dagger g_k.
$$

两者一般不同。只有当 $\eta$ 很小，proximal 更新才有一阶近似

$$
\left(H_k+\frac1\eta F_k\right)^\dagger g_k
=
\eta F_k^\dagger g_k+O(\eta^2).
$$

若再采用 F.5 的局部二次假设 $H_k\approx F_k\approx A$，则你的外层更新是

$$
\theta_{k+1}-\theta_k
\approx
-\frac{\eta}{1+\eta}F_k^\dagger g_k,
$$

即有效 NG 步长是

$$
\gamma=\frac{\eta}{1+\eta},
$$

而不是 $\eta$。

它与 F.5 的 NGD 动力学如何对应？在 F.5 的冻结局部模型

$$
L(\Delta\theta)
=
\frac12\Delta\theta^\top A\Delta\theta-b^\top\Delta\theta+\mathrm{const},
$$

且 $F=A$ 时，若每个 outer iteration 精确解你的 KL-proximal 子问题，则

$$
\Delta\theta_{k+1}
=
\frac{1}{1+\eta}\Delta\theta_k
+
\frac{\eta}{1+\eta}A^\dagger b.
$$

这正好是 F.5 中冻结 Fisher 的 NGD 递推

$$
\Delta\theta_{k+1}
=
(1-\gamma)\Delta\theta_k+\gamma A^\dagger b
$$

在 $\gamma=\eta/(1+\eta)$ 时的形式。

所以结论很精确：

- 多 PPO epoch + IS + 锚定 KL：是对冻结-rollout 的 KL-proximal OPD 的 SGD 近似。
- 若内层求解充分准确，且 F.5 的局部二次/Fisher 近似成立：外层迭代满足 F.5 的 NGD 动力学，但步长是 $\eta/(1+\eta)$。
- 有限个 SGD epoch 时：它只是朝该 proximal 点移动；不会严格遵循 F.5 的 NGD 递推。
- 若想让外层严格对应步长 $\eta$ 的 NGD：应线性化 OPD 项，并精确求解 KL-MD 子问题，或直接求解 $F_kv=g_k$ 后更新 $\theta\leftarrow\theta-\eta v$。
