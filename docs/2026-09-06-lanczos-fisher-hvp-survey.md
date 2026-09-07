# OPD 中如何计算 Fisher/Hessian 谱：HVP、FVP 与 Lanczos

> 更新：2026-09-07。本版只保留对 OPD 决策真正有用的论文、代码和资源预算；其余工作列为简短参考。

## 0. 先给结论：现在不要把 HVP 接进训练

HVP/Lanczos 是**定期冻结态探针**，不是每步训练指标。对当前 1--2B 级模型，推荐在 step 30/50 保存 checkpoint，用固定的 4--16 条 rollout、每条 128--512 个 response token 做 10--20 步 Lanczos。第一次探针的计算预算大致是 10--20 次二阶 matvec；在 activation checkpointing 和 microbatch 下，峰值显存通常约为同一 probe batch 一次训练 forward+backward 的 1.5--3 倍，而不是参数平方级增长。显式 Hessian 仍然不可行。

### 对我们最重要的三个选择

1. **第一版测 GGN/Fisher，不测 full reverse-KL Hessian。** GGN 是半正定，Lanczos 更稳定，且正好对应我们理论中的 (A=J_z^\top F_zJ_z)。
2. **第一版只求 top-10 主方向。** 不要一开始做完整谱密度；我们当前问题是“EffOPD 更新方向是否落在 OPD 任务曲率主空间”，不是估计全模型 Hessian density。
3. **必须保存任务投影。** 仅有 (lambda_i) 没有意义；同时保存梯度投影 (b_i)、更新投影 (u_i)、误差投影 (x_i) 和跨 microbatch 方差。

## 0.1 显存预算（按我们当前 OPD 实现）

当前 actor 的 `_compute_old_log_prob` 已经执行 `calculate_entropy=True`，并把逐 token entropy 存在 `batch.batch["entropys"]`；teacher worker 已经同时保存 `old_log_probs` 和 `teacher_log_probs`。因此新增的 OPD diagnostics 只做 `(B,T)` 张量上的减法、绝对值、平方和 mask-reduction：不会保存 logits，也不会建立词表矩阵。额外显存是少量 (B\times T) 临时 buffer，训练时间主要是几次轻量 reduction，预期远小于 1%。

严格同批次的 `teacher_nll_mean`、`student_nll_mean` 和 `student_entropy_mean` 也复用已有张量，不触发 forward。只有 full-vocabulary KL、GGN-HVP 和 Hessian-HVP 会显著增加成本：前者需要保留或重算 vocab logits，后两者需要二阶 autograd 或有限差分的重复 forward/backward。

对 1--2B 模型，保守地按一次训练 microbatch 的峰值显存记账：

| 探针                        |         额外显存（相对一次普通 forward+backward） |                         计算代价 |
| --------------------------- | ------------------------------------------------: | -------------------------------: |
| 10 步 GGN/Fisher Lanczos    | 约 0.5--2 倍，取决于 checkpoint 和 probe token 数 |                     10 次 matvec |
| 20 步 GGN/Fisher Lanczos    |                                      约 0.5--2 倍 |                     20 次 matvec |
| full reverse-KL Hessian-HVP |                                      约 1.5--3 倍 | 每次需保留一阶反传图并再反传一次 |
| 有限差分 HVP                |                                   约 1 倍峰值显存 |         每次 matvec 两次梯度计算 |

这些倍数取决于 FSDP shard、序列长度、microbatch 和是否保存 Ritz 向量；不是硬件保证。第一版应把 probe 限制为 4--16 条序列、每条 128--512 个 response token，并在独立 checkpoint 脚本中运行。

本文回答一个具体问题：如果要验证 OPD 的谱吸收理论，怎样实际估计

$$
A=\mathbb E_c[J_c^\top F_cJ_c]
$$

的主特征值和特征向量？重点说明 Hessian-vector product（HVP）、Fisher-vector product（FVP）、Generalized Gauss--Newton（GGN）和 Lanczos 的关系，以及它们在 OPD 上的可执行实现。

## 1. 先区分四个矩阵

### 1.1 Hessian

对训练目标 $L(\theta)$，Hessian 是

$$
H(\theta)=\nabla_\theta^2L(\theta).
$$

它是局部二阶曲率的最直接定义，但在非凸网络中可能有负特征值。若 $L$ 是 OPD 的 reverse-KL 目标，$H$ 还包含网络 Jacobian 的二阶导数项，以及 on-policy 状态分布固定时的经验目标曲率。

### 1.2 Fisher

策略分布的真 Fisher 为

$$
F(\theta)=\mathbb E_{c,y\sim\pi_\theta}
\left[\nabla_\theta\log\pi_\theta(y\mid c)
\nabla_\theta\log\pi_\theta(y\mid c)^\top\right].
$$

它是半正定的，代表概率分布流形上的局部 KL 度量。真 Fisher 的期望分布是模型自身的 $y\sim\pi_\theta$，而不是只对训练标签或已采样标签平方梯度。

### 1.3 GGN 与 OPD 的 $A$

在 logits 线性化

$$
z_\theta(c)\simeq z_0(c)+J_c\Delta\theta
$$

以及输出 KL 二次化后，OPD 的局部矩阵为

$$
A=\mathbb E_c[J_c^\top F_cJ_c].
$$

这正是以输出空间 Fisher $F_c$ 为损失曲率的 GGN 矩阵。对 softmax 交叉熵，输出 Fisher 与交叉熵 Hessian 相同，因此 GGN、模型 Fisher 和局部 Hessian 在满足相应条件时一致或非常接近[^martens14]。

### 1.4 empirical Fisher 不是这里想要的矩阵

常见的 empirical Fisher 是

$$
\bar F=\frac1B\sum_{b=1}^B g_b g_b^\top,
$$

其中 $g_b$ 是当前样本的梯度。它计算方便，但在模型已经接近标签/教师分布时，逐样本梯度并不会自动趋近真 Fisher；因此它不能无条件替代 $F$ 或 GGN[^kunstner19]。对于 OPD，若我们要验证“剩余误差的曲率谱”，优先使用固定状态上的 GGN/Fisher，而不是把 batch 梯度外积直接称为 Fisher。

## 2. 为什么不显式构造矩阵

1.5B 模型的参数维度为 $P\sim10^9$，显式保存 $P\times P$ 矩阵不可能。Lanczos 只需要一个黑盒函数

$$
v\mapsto Av,
$$

即矩阵-向量乘法。若能计算 $Av$，就可以在 $m\ll P$ 次乘法后得到一个 $m\times m$ 的三对角矩阵，其特征值近似 $A$ 的极值和谱密度。

HVP 的经典自动微分实现来自 Pearlmutter：先算一阶梯度 $g=\nabla L$，再对标量 $g^\top v$ 求一次梯度：

$$
\operatorname{HVP}(v)
=\nabla_\theta(g^\top v)=H v.
$$

它的时间复杂度与一次梯度计算同阶，内存主要是保留一阶反向图，而不是 $O(P^2)$[^pearlmutter94]。

## 3. HVP、FVP 和 GGN-vector 的具体计算

### 3.1 Hessian-vector product

给定固定 batch 上的标量损失 `loss` 和同形状向量 `v`：

```python
grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
dot = sum((g * x).sum() for g, x in zip(grads, v))
hvp = torch.autograd.grad(dot, params, retain_graph=False)
```

数学上这是

$$
\nabla_\theta\big(\nabla_\theta L(\theta)^\top v\big).
$$

若使用 FSDP/张量并行，需要保证 `v` 和参数梯度使用同一分片布局；否则得到的只是 shard-local 近似而不是全局 HVP。大模型上通常采用 flat parameter buffer、通信聚合，或在每个 shard 上使用与训练相同的 all-reduce。

### 3.2 输出空间 GGN-vector

对于

$$
A=J^\top C J,
$$

其中 $J$ 是 logits 对参数的 Jacobian，$C$ 是输出空间曲率，计算 $Av$ 可以按三步完成：

1. JVP：计算 $u=Jv$，即沿参数方向 $v$ 的 logits 变化；
2. 输出曲率：计算 $w=Cu$；
3. VJP：计算 $J^\top w$。

因此

$$
Av=J^\top(C(Jv)).
$$

对 softmax 分布 $p$，输出 Fisher 为

$$
C=\operatorname{Diag}(p)-pp^\top,
$$

所以

$$
Cu=p\odot u-p(p^\top u).
$$

这个乘法不需要建立词表大小乘词表大小的矩阵。

在实现上，可以使用 forward-mode JVP + reverse-mode VJP；若框架对大模型 JVP 支持不稳定，也可以用二阶自动微分直接计算 GGN/Hessian-vector。对 OPD 的 logits KL，建议优先固定一批状态和 teacher logits，然后在该固定 batch 上重复调用同一个 `matvec(v)`。

### 3.3 Fisher-vector product 的两种口径

真 Fisher 的矩阵乘法可以写成

$$
Fv=\mathbb E_{c,y\sim p_\theta}
\left[s_{c,y}(s_{c,y}^\top v)\right],
\qquad
s_{c,y}=\nabla_\theta\log p_\theta(y\mid c).
$$

逐 token 构造 score 会非常昂贵，但对分类/softmax 输出可以使用 GGN 形式：先计算 logits JVP，再乘输出 Fisher，再做 VJP。此时得到的是模型分布下的 Fisher/GGN，而不是 empirical Fisher。

对 OPD 的 reverse-KL 目标，若局部展开点在 teacher 附近，输出曲率应使用 teacher 分布 $p_T$；若为了构造固定的训练前近似而使用学生初始分布 $p_0$，必须明确这是一个 frozen-base approximation，而不是严格的当前 Fisher。

## 4. Lanczos 算法如何把 matvec 变成特征谱

### 4.1 三项递推

给定对称矩阵 $A$ 的 matvec，从单位随机向量 $q_1$ 开始：

$$
\beta_0=0,\qquad q_0=0.
$$

第 $j$ 步计算

$$
z=Aq_j-\beta_{j-1}q_{j-1},
$$

$$
\alpha_j=q_j^\top z,
\qquad
z\leftarrow z-\alpha_jq_j,
$$

$$
\beta_j=\|z\|,
\qquad
q_{j+1}=z/\beta_j.
$$

得到三对角矩阵

$$
T_m=
\begin{pmatrix}
\alpha_1&\beta_1&&\\
\beta_1&\alpha_2&\ddots&\\
&\ddots&\ddots&\beta_{m-1}\\
&&\beta_{m-1}&\alpha_m
\end{pmatrix}.
$$

对 $T_m$ 做普通特征分解：

$$
T_m y_i=\hat\lambda_i y_i.
$$

则 $\hat\lambda_i$ 是 Ritz 特征值，近似 $A$ 的特征值；对应的 Ritz 向量近似为

$$
\hat u_i=Q_my_i,
\qquad Q_m=[q_1,\ldots,q_m].
$$

### 4.2 为什么它能找到主方向

Lanczos 本质上在 Krylov 子空间

$$
\mathcal K_m(A,q_1)=\operatorname{span}\{q_1,Aq_1,A^2q_1,\ldots,A^{m-1}q_1\}
$$

中寻找谱信息。若随机起点对某个特征向量有非零投影，反复乘 $A$ 会放大大特征值方向，因此极值特征值通常较快收敛。谱中间部分的分辨率较差，需要更多步数或随机起点。

### 4.3 数值稳定性

有限精度下，理论上正交的 $q_j$ 会失去正交性，导致重复 Ritz 值（ghost eigenvalues）。实践中可使用：

- full reorthogonalization：最稳，但需要保存所有 $q_j$；
- selective/partial reorthogonalization：大规模问题更省内存；
- 多个随机起点的 stochastic Lanczos quadrature（SLQ）：适合谱密度而不是只找 top eigenvectors。

Ghorbani et al. 将分布式 Lanczos 与 HVP 结合，用于大规模网络 Hessian 谱和谱密度估计；Papyan 的工作使用 stochastic Lanczos quadrature 研究深网 Hessian 的 bulk、outlier 和训练期演化[^ghorbani19][^papyan20]。

## 5. SLQ 如何估计整个谱密度

如果目标不是前几个特征向量，而是谱密度 $\rho(\lambda)$，对随机向量 $z$ 使用二次型

$$
z^\top f(A)z
$$

的 Lanczos 近似。对每个随机起点得到 $T_m$ 后，将其特征值 $\tilde\lambda_j$ 和权重 $|e_1^\top y_j|^2$ 组成离散测度：

$$
z^\top f(A)z
\approx
\|z\|^2\sum_{j=1}^m
|e_1^\top y_j|^2 f(\tilde\lambda_j).
$$

对多个随机起点平均即可近似

$$
\frac1P\operatorname{Tr}f(A),
$$

选择 $f$ 为高斯核或直方图核就能得到谱密度估计。需要注意：SLQ 估计的是 trace-weighted density；我们的吸收率模型需要的是由初始误差振幅 $a_i$ 加权的谱，不是裸 Hessian 的 trace 谱。后者只能说明“模型有哪些曲率”，不能单独说明“剩余误差占据哪些曲率”。

## 6. 在 OPD 上的完整计算方案

### 6.1 固定目标

先使用 off-policy/frozen-state 设置，固定一批上下文和 teacher logits。这样 $A$ 在一个测量窗口内有明确含义，避免 on-policy 状态漂移把谱变化和样本变化混在一起。

### 6.2 定义 matvec

推荐定义

$$
\operatorname{matvec}(v)
=\frac1{|\mathcal S|}\sum_{c\in\mathcal S}
J_c^\top\left[F_c(J_cv)\right].
$$

在 softmax 输出上，先做

$$
u_c=J_cv,
\qquad
w_c=p_c\odot u_c-p_c(p_c^\top u_c),
$$

再做 $J_c^\top w_c$。如果直接用 loss HVP，则要记录所用 loss 和展开点，以便区分 Hessian 与 GGN。

### 6.3 运行 Lanczos

```python
def lanczos(matvec, dim, steps, device, reorthogonalize=True):
    q = torch.randn(dim, device=device)
    q = q / q.norm()
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), device=device)
    basis, alpha, beta = [], [], []

    for _ in range(steps):
        basis.append(q)
        z = matvec(q) - beta_prev * q_prev
        a = torch.dot(q, z)
        z = z - a * q
        if reorthogonalize:
            for old in basis:
                z = z - torch.dot(old, z) * old
        b = z.norm()
        alpha.append(a)
        if b < 1e-10:
            break
        beta.append(b)
        q_prev, q = q, z / b
        beta_prev = b

    T = torch.diag(torch.stack(alpha))
    if len(beta):
        off = torch.stack(beta[: len(alpha) - 1])
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    eigvals, eigvecs = torch.linalg.eigh(T)
    Q = torch.stack(basis, dim=1)
    eigvecs_full = Q @ eigvecs
    return eigvals, eigvecs_full, T
```

伪代码中的 full reorthogonalization 适合小的 $m$，例如 $m=50$--$150$；大规模运行应使用 selective reorthogonalization 或只保留三对角系数。

### 6.4 记录什么

每个测量 checkpoint 至少记录：

- top-$k$ Ritz 特征值 $\hat\lambda_i$；
- Ritz 向量与下一 checkpoint 主子空间的 cosine similarity；
- $\hat u_i^\top\delta\theta$，即初始参数位移在该方向上的振幅；
- HVP/FVP 残差 $\|A\hat u_i-\hat\lambda_i\hat u_i\|$；
- 多随机起点之间的特征值方差；
- 固定状态集和当前 on-policy 状态集上的谱差异。

## 7. 如何与谱吸收理论连接

### 7.1 从 Ritz 模式预测 KL 曲线

若在 step $t_0$ 测得 $\hat\lambda_i$，并估计参数位移振幅

$$
\hat x_i=\hat u_i^\top(\theta_{t_0}-\theta^*),
$$

则局部二次 KL 的预测为

$$
d^{\mathrm{KL}}(t)
\approx
\frac12\sum_{i=1}^k
\hat\lambda_i\hat x_i^2
e^{-2\eta\hat\lambda_i(t-t_0)}
d_{\mathrm{tail}}(t).
$$

注意 KL 的速率是 $2\eta\lambda_i$，因为 KL 是平方型误差；L1 代理则更接近 $e^{-\eta\lambda_i t}$，不能直接共用同一组振幅。

### 7.2 验证“早期 low-rank lock-in”

论文中的 low-rank update 现象对应两个可分开的检验：

1. 更新矩阵的 top-$k$ SVD 子空间是否与 $A$ 的 top-$k$ Ritz 子空间对齐；
2. $A$-谱中的驱动力振幅 $b_i=\hat u_i^\top b$ 是否集中在这些方向。

只有两者都成立，才能把“参数更新低秩”解释成“OPD 主要吸收少数曲率模式”。如果欧氏 SVD 很集中但 $A$-谱不集中，低秩只是参数化效应；如果 $A$-谱集中但 $b_i$ 很小，说明这些是模型有曲率的方向，却不是当前教师任务真正激活的方向。

### 7.3 验证吸收率下降

对于加权模式分布 $w_i(t)$，理论给出

$$
\frac{dv}{dt}\simeq-\eta^2\operatorname{Var}_{w(t)}(\lambda).
$$

Lanczos 提供的是候选 $\lambda_i$；要得到理论中的 $w_i(t)$，还必须用残差振幅或 KL 权重重加权。因此只测 Hessian 最大特征值并不能解释吸收率曲线，至少需要“特征值 + 任务相关振幅”两部分。

## 8. 计算成本与工程特性

### 8.1 成本

一次 HVP/FVP 通常约为一次到两次 forward/backward 的成本，具体取决于实现和是否保留图。$m$ 步 Lanczos 需要约 $m$ 次 matvec；$r$ 个随机起点的 SLQ 需要约 $rm$ 次。对 1.5B 模型，$m=100$ 已经是明显的额外计算，不适合每个训练 step 执行，更适合在少量 checkpoint 上做测量。

### 8.2 优点

- 不需要显式保存 Hessian/Fisher；
- 只访问 matvec，可以复用自动微分和分布式并行；
- top eigenvalues 对谱隙较大的情形收敛快；
- 可以同时获得主方向、曲率和谱密度；
- 对固定状态集尤其适合，可直接检验准静态谱假设。

### 8.3 局限

- Lanczos 对中间谱和极小特征值分辨率较弱；
- mini-batch matvec 是随机的，破坏严格对称性，必须固定 batch 或做多 batch 平均；
- bf16 的二阶导数容易出现数值噪声，建议至少用 fp32 累积三对角系数；
- 非凸 Hessian 可能有负特征值，而 Fisher/GGN 理论上半正定；两者结果不能混用；
- 参数矩阵 SVD 的方向与 Fisher 特征向量不一定对齐；
- on-policy 时 $A_t$ 随状态分布和参数变化，单次 Lanczos 只描述一个局部窗口。

## 9. 相关工作的定位

- Pearlmutter (1994) 给出了不显式构造 Hessian 的精确 Hessian-vector 自动微分方法，是 HVP 的基础[^pearlmutter94]。
- Martens (2020) 系统讨论了 Hessian、GGN、Fisher 和自然梯度之间的关系，并强调 Fisher 的正定性、damping 和病态方向问题[^martens14]。
- Ghorbani, Krishnan & Xiao (2019) 将分布式 Lanczos 用于大规模神经网络 Hessian 特征值和谱密度估计，指出运行时间主要由 HVP 数量决定[^ghorbani19]。
- Papyan (2020) 使用随机 Lanczos 研究深网 Hessian 的 bulk、outlier 和训练过程中的谱演化，为“曲率谱会随训练变化”提供了系统实证[^papyan20]。
- Martens & Grosse (2015) 的 K-FAC 不直接求谱，而是用 Kronecker 结构近似 Fisher 逆；它与这里的关系是：K-FAC 试图便宜地近似预条件器，而 Lanczos 试图测量预条件前的谱[^martens-grosse-15]。
- Kunstner, Hennig & Balles (2019) 说明 empirical Fisher 与真 Fisher 的差异，提醒我们不能把 batch 梯度外积当作理论中的 Fisher[^kunstner19]。

## 10. 针对 OPD 的建议配置

最小而可靠的第一版测量流程是：

1. 选择 off-policy 的固定 64 条轨迹和固定 teacher logits；
2. 在 step 30、100、200 三个 checkpoint 上定义同一个 GGN/Fisher matvec；
3. 每个 checkpoint 用 2--4 个随机起点、每个 50--100 步 Lanczos；
4. 保存 top-10 Ritz 对、残差范数和随机起点方差；
5. 用 checkpoint 位移投影估计 $\hat x_i$，用 KL 权重构造加权谱；
6. 预测 $d(t)$ 和 $v_t$，与实际曲线及 NNLS 反演比较；
7. 再把同一测量移到 on-policy checkpoint，比较 $\lambda_i(t)$ 和主子空间的漂移。

这样可以依次回答三个问题：

- OPD 的更新低秩是否真的对应 Fisher/GGN 的低维主子空间？
- 这些主方向是否承载了教师残差，而不仅仅是模型本身的高曲率？
- 测得的谱和振幅能否预测吸收率下降及其长尾？

如果第三个问题得到肯定，才有充分依据把 Spectral EffOPD 从一个启发式 checkpoint 外推器升级为真正的低秩谱预条件更新。

## 11. 2025--2026 年 LLM 相关工作的补充

这一节专门补充近期与“在 Transformer/LLM 上测曲率谱或使用完整二阶信息”直接相关的工作。它们的重要性不只是提供了更多论文，而是改变了一个实际前提：**Hessian 谱测量已经从小模型诊断工具，逐渐变成可以在十亿到千亿参数模型上运行的系统组件。**

### 11.1 HessFormer：70B 级模型上的分布式 SLQ

Granziol 的 HessFormer（2025，arXiv:2505.11564）把 Pearlmutter HVP 和 stochastic Lanczos quadrature 接到 Hugging Face/PyTorch 训练栈中，并在单节点多 GPU 上测量 DeepSeek 70B 的 Hessian 谱[^hessformer25]。其贡献主要是系统工程：保持数据并行、张量并行和 pipeline 拆分，不显式汇集完整参数矩阵，而是在每次 Lanczos 迭代中分布式计算 HVP。

对我们最有用的三个经验是：

1. Lanczos 的瓶颈几乎完全是 HVP 次数；三对角矩阵构造和后处理相对便宜。
2. 不同数据子集会改变谱密度，因而 OPD 的谱测量必须固定状态集、序列长度和 batch 采样口径。
3. 低迭代数会出现 Ritz 值重合或“ghost”结构；不能把 SLQ 的每个离散峰直接当成 Hessian 的真实重数。

HessFormer测的是经验 Hessian，而不是 OPD 理论中的任务加权谱。它证明了“能测”，但不等于已经证明某个谱能解释吸收率。

### 11.2 Foundation-scale Hessian Spectral Analysis：FSDP 下的有限差分 HVP

2026 年的 _Hessian Spectral Analysis at Foundation Model Scale_（arXiv:2602.00816）进一步把谱测量推进到最高约 100B 参数，并针对 FSDP 提出 shard-local central finite-difference HVP[^foundationhessian26]。它不依赖二阶 autograd，而是用

$$
Hv\approx
\frac{\nabla L(\theta+\varepsilon v)-
\nabla L(\theta-\varepsilon v)}{2\varepsilon}.
$$

这带来一个重要的工程/数值权衡：截断误差约为 $O(\varepsilon^2)$，浮点舍入和梯度噪声则随 $1/\varepsilon$ 放大。论文给出了有限差分误差和 Lanczos 三对角扰动的分析，并特别讨论 fp32 与 bf16 的可用区间。其主要结论包括：

- shard-local HVP 可以避免 FSDP 的全参数 gather；
- Lanczos/SLQ 的额外成本主要是重复 HVP，向量运算和标量 all-reduce 较小；
- 在中等规模 LLM 上，常用的 block-diagonal 曲率近似可能出现数量级的相对误差和较差的方向对齐。

这对我们的 OPD 方案有直接警示：如果我们只测每层独立 Hessian，就可能漏掉跨层耦合；如果只用 bf16 二阶结果，则必须报告有限差分/自动微分误差，而不能把小特征值的抖动解释成真实谱尾。

### 11.3 Full Gauss--Newton for LLMs：理想二阶上限

_The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss--Newton_（2025，arXiv:2510.09378）在最多约 150M 参数的 Transformer 上直接研究 full Gauss--Newton 预条件，并将其视为现有优化器的迭代复杂度上限探针[^fullgn25]。论文报告：相对于 SOAP 等强基线，full GN 在其设置下可达到约 5.4 倍的迭代减少；只忽略跨层信息的 layerwise GN 已接近 full GN。

这项工作的含义不是“应该立刻用 full GN 训练 1.5B OPD”，而是提供两个可检验假设：

1. LLM 训练中最有价值的二阶信息可能主要来自每层曲率，而不是全部跨层项；
2. 当前 SOAP、Muon 等方法与理想 GN 之间仍存在明显的谱预条件差距。

对 OPD 来说，layerwise GN 是比 full Fisher 更现实的第一步：它可以检测“中间层 MLP 的高边际收益”究竟是模块驱动力 $b_m$ 大，还是仅仅因为该模块曲率更高。

### 11.4 Why Transformers Need Adam：block heterogeneity

_Why Transformers Need Adam: A Hessian Perspective_（arXiv:2402.16788，NeurIPS 2024）使用 SLQ 比较 Transformer 和 CNN 的 Hessian block 谱[^transformeradam24]。论文的核心概念是 **block heterogeneity**：不同参数块的 Hessian 谱差异很大。单一全局 learning rate 的 GD 在这种情况下受到最坏条件数限制，而 Adam 的坐标/块级尺度调整可以缓解这一问题。

这与我们的模型形成一个非常直接的对应：

$$
\text{全局标量 }\eta
\quad\longrightarrow\quad
\text{块/方向相关的有效 }\eta_i.
$$

它支持把 Spectral EffOPD 看成动态 learning rate 的谱推广，而不是单纯的 checkpoint 技巧。同时，论文报告 block heterogeneity 在预训练后会减弱，说明 $A$ 的谱不是静态常数；OPD 的 checkpoint 谱测量必须记录时间，而不能只测初始化或最终模型。

### 11.5 结构化 Fisher 优化器与低秩扩展

ICLR 2026 的 _Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension_ 直接从结构化 Fisher 近似出发设计 LLM 优化器[^structuredfisher26]。这条路线与我们提出的低秩谱更新非常接近，但目标不同：该工作重点是以可控内存和计算代价近似 Fisher 预条件器；我们的目标是用少量主方向解释 OPD 的吸收率和 checkpoint 外推。

从方法上可以借鉴两点：

- 先用结构化近似捕捉大部分曲率，再用低秩修正补上最重要的全局耦合；
- 评价预条件器时不能只看训练 loss，还要看方向对齐和预条件后谱的条件数。

这说明 OPD 的谱方法可以有一个更实际的折中形式：

$$
P\approx P_{\mathrm{layerwise/structured}}
+U_kC_kU_k^\top,
$$

即“便宜的块预条件 + 少量全局低秩修正”，而不必从零构造 full Fisher inverse。

### 11.6 近期工作的共同边界

把这些论文放在一起后，可以得到四条共识：

1. **测谱可行，但测到的谱依赖数据和数值口径。** HessFormer 与 foundation-scale 工作都强调 batch、有限差分步长、精度和重正交化。
2. **Hessian、GGN、Fisher 不能混称。** full GN 的正定预条件与非凸 Hessian 的负特征值有不同用途。
3. **块对角近似并非总是可靠。** LLM 的 block heterogeneity 和 foundation-scale 的 block-diagonal 误差结果都支持先测跨块方向对齐，再决定是否采用 layerwise 近似。
4. **谱密度不等于任务谱。** SLQ 的 trace-weighted density必须与 OPD 教师残差振幅 $b_i$ 或 KL 权重结合，才可用于解释 $v_t$。

对我们的实验设计，最稳妥的优先级因此是：先在固定状态上用 GGN/Fisher matvec 测 top Ritz 方向和残差投影，再用 SLQ 估计谱密度，最后才讨论用该谱做预条件或 EffOPD 外推。

### 参考文献

[^pearlmutter94]: B. A. Pearlmutter. _Fast Exact Multiplication by the Hessian_. Neural Computation 6(1):147--160, 1994. [doi:10.1162/neco.1994.6.1.147](https://doi.org/10.1162/neco.1994.6.1.147)

[^martens14]: J. Martens. _New Insights and Perspectives on the Natural Gradient Method_. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^ghorbani19]: B. Ghorbani, S. Krishnan, Y. Xiao. _An Investigation into Neural Net Optimization via Hessian Eigenvalue Density_. ICML 2019. [arXiv:1901.10159](https://arxiv.org/abs/1901.10159)

[^papyan20]: V. Papyan. _The Full Spectrum of Deepnet Hessians at Scale: Dynamics with SGD Training_. IEEE Transactions on Information Theory, 2020.

[^martens-grosse-15]: J. Martens, R. Grosse. _Optimizing Neural Networks with Kronecker-factored Approximate Curvature_. ICML 2015. [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)

[^kunstner19]: F. Kunstner, P. Hennig, L. Balles. _Limitations of the Empirical Fisher Approximation for Natural Gradient Descent_. NeurIPS 2019. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)

[^hessformer25]: D. Granziol. _HessFormer: Hessians at Foundation Scale_. arXiv 2025. [arXiv:2505.11564](https://arxiv.org/abs/2505.11564)

[^foundationhessian26]: _Hessian Spectral Analysis at Foundation Model Scale_. arXiv 2026. [arXiv:2602.00816](https://arxiv.org/abs/2602.00816)

[^fullgn25]: _The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss--Newton_. arXiv 2025. [arXiv:2510.09378](https://arxiv.org/abs/2510.09378)

[^transformeradam24]: _Why Transformers Need Adam: A Hessian Perspective_. arXiv 2024. [arXiv:2402.16788](https://arxiv.org/abs/2402.16788)

[^structuredfisher26]: W. Gong, M. Scetbon, C. Ma. _Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension_. ICLR 2026. [OpenReview](https://openreview.net/forum?id=KUFZXdem5R)
