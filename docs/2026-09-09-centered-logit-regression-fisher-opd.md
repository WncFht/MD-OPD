# CLR + VGD for OPD：从 Reverse-KL 的 Fisher 几何到采样与 Top-k 训练

**日期**：2026-09-09  
**主题**：固定 teacher、固定 context 的 off-policy / on-policy distillation，centered-logit regression（CLR）、vanilla gradient descent（VGD）、Fisher natural gradient（NGD）以及 sampled-token 和 top-k 估计。

## 摘要

在固定 context 下，teacher 给出一个 categorical 分布 $q$，student 给出

$$
p_\theta=\operatorname{softmax}(z_\theta).
$$

OPD 常用的 sampled token reward 是

$$
r(a)=\log q(a)-\log p_\theta(a).
$$

当 token 从 student 分布采样时，它在期望上对应 reverse KL：

$$
\mathbb E_{a\sim p_\theta}[r(a)]
=-D_{\mathrm{KL}}(p_\theta\Vert q).
$$

问题在于，普通 logit-space vanilla gradient 会把 reverse-KL 的梯度再乘一次 categorical Fisher，因此高概率方向快、低概率方向慢。理想的 output-space natural-gradient 更新则消掉这层 Fisher，令所有 log-odds error 以同一个比例收缩。

这引出 centered-logit regression：

$$
L_{\mathrm{CLR}}(z,u)
=\frac12\left\Vert P(z-u)\right\Vert_2^2,
\qquad
P=I-\frac1K\mathbf 1\mathbf 1^\top.
$$

它的输出 logit 梯度正好是

$$
\nabla_zL_{\mathrm{CLR}}=P(z-u),
$$

因此在完整词表、直接优化 logits 的理想设定下：

$$
\boxed{
\text{CLR + VGD}
=
\text{reverse-KL + output-space NGD}
}.
$$

这个目标的基本思想有 direct logit regression、高温知识蒸馏和 pairwise logit matching 等先例；最近的 Concrete Score Distillation（CSD）[^csd26] 更明确地用加权 pairwise logit differences 处理 softmax 平滑和 logit shift invariance。本文的重点不是把 CLR 宣称为一个完全孤立的新 loss，而是把它放回 OPD 的 Fisher 几何中，说明它为什么是 reverse-KL 的 naturalized vector field，以及在 sampled-token、top-k 和神经网络参数训练中如何实现。

## 1. 问题设定

### 1.1 固定 teacher 与固定 context

固定一个 context，teacher 的 logit 和分布分别记为 $u$、$q$，student 的 logit 和分布分别记为 $z$、$p$：

$$
q_i=\frac{\exp(u_i)}{\sum_j\exp(u_j)},\qquad
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
$$

本文首先只讨论输出 logits 直接被优化的理想情况。也就是说，$z$ 是优化变量，teacher $u$ 永远冻结，每一步都在同一个 context 上重复更新。

这个设定很小，但它把三个问题分开了：

1. reverse-KL 本身的几何曲率；
2. stochastic token estimator 的偏差和方差；
3. 神经网络 Jacobian 和 optimizer 对输出动力学的进一步改变。

真实 OPD 还会改变 student 生成的 prefix，因此 context 分布也会漂移。本文把这种状态分布问题放到最后讨论；如果要研究“固定目标上的优化速度”，必须先看现在这个 frozen-context 模型。

### 1.2 OPD reward 与 reverse KL

令

$$
r_i=\log q_i-\log p_i.
$$

则

$$
\mathbb E_{i\sim p}[r_i]
=\sum_i p_i\log\frac{q_i}{p_i}
=-D_{\mathrm{KL}}(p\Vert q).
$$

因此，用 student 采样 token 并最大化 reward，在期望上是在最小化 reverse KL。

这里的 KL 方向非常重要。传统分类知识蒸馏通常写 teacher-to-student 的 forward KL，而 OPD / MiniLLM / GKD 讨论的是 student-to-teacher 的 reverse KL。MiniLLM 将 reverse KL 作为生成模型蒸馏目标，GKD 则把 supervised 和 on-policy sequence distillation 统一到可选择 divergence 的框架中[^minillm24][^gkd24]。

## 2. Softmax 的 gauge invariance 与 Fisher

### 2.1 Logit 的公共平移没有意义

对任意常数 $c$，

$$
\operatorname{softmax}(z+c\mathbf 1)=\operatorname{softmax}(z).
$$

所以概率分布只依赖 logits 的 quotient space：

$$
\mathbb R^K/\operatorname{span}\{\mathbf 1\}.
$$

如果 teacher 和 student 的分布相同，正确条件不是 $z=u$，而是

$$
z=u+c\mathbf 1.
$$

这也是 raw logit MSE 的一个问题。它会人为选择 $c=0$ 的 gauge，并惩罚实际上不影响分布的公共平移。

### 2.2 Categorical Fisher

对 softmax logits，Fisher 矩阵为

$$
F_p=\operatorname{diag}(p)-pp^\top.
$$

它满足

$$
F_p\mathbf 1=0,\qquad
\operatorname{range}(F_p)=\mathbf 1^\perp.
$$

零特征值正是公共 logit 平移方向。Fisher 只在真实的 probability tangent space 上提供度量。

### 2.3 Reverse-KL 的 logit 梯度

定义 log-probability error

$$
\ell_i=\log p_i-\log q_i.
$$

因为

$$
\ell_i=(z_i-u_i)+c
$$

其中 $c$ 与 $i$ 无关，所以

$$
F_p\ell=F_p(z-u).
$$

对 reverse KL 求导：

$$
\begin{aligned}
\nabla_zD_{\mathrm{KL}}(p\Vert q)
&=F_p\ell \\
&=F_p(z-u).
\end{aligned}
$$

这条式子解释了普通 VGD 的动力学：它并不是直接沿着 logit error 更新，而是沿着 Fisher-filtered error 更新。

如果用 Fisher pseudoinverse 做 output-space natural gradient，则

$$
F_p^+\nabla_zD_{\mathrm{KL}}(p\Vert q)
=F_p^+F_p(z-u)
=P(z-u).
$$

因此 natural gradient 只消除公共平移之外的 Fisher 曲率，不会试图更新不可辨识的 gauge 方向。

## 3. 三种 logit 动力学

### 3.1 Reverse-KL + vanilla gradient

令 $e=z-u$。普通 gradient descent 为

$$
e^+=e-\alpha F_pe.
$$

接近 teacher 时 $p\approx q$，可以线性化为

$$
e^+\approx(I-\alpha F_q)e.
$$

如果 $F_q$ 在 quotient space 上的特征值是 $\lambda_j$，每个模式的收缩因子就是

$$
1-\alpha\lambda_j.
$$

因此 vanilla gradient 有多个时间尺度。小 Fisher eigenvalue 的方向，通常对应低概率或 rare-token 方向，会非常慢。

这与二分类的例子一致。令 $x=z_1-z_2$、$y=u_1-u_2$，则 vanilla 更新近似为

$$
x^+=x+2\alpha p(1-p)(y-x).
$$

当 $p$ 接近 $0$ 或 $1$ 时，$2p(1-p)$ 接近零，更新几乎停止。

### 3.2 Reverse-KL + output natural gradient

natural update 为

$$
z^+=z-\eta P(z-u).
$$

忽略一个公共常数，它等价于

$$
z^+\equiv(1-\eta)z+\eta u.
$$

于是每个 logit pair 的误差都以同一比例收缩：

$$
\left[(z_i-z_j)-(u_i-u_j)\right]^+
=(1-\eta)
\left[(z_i-z_j)-(u_i-u_j)\right].
$$

对应的概率更新是 geometric mixture：

$$
p_i^+
\propto
p_i^{1-\eta}q_i^\eta.
$$

当 $0<\eta<2$ 时，线性 quotient dynamics 稳定；当 $\eta=1$ 时，输出分布一步到达 teacher 分布。

### 3.3 CLR + vanilla gradient

定义

$$
L_{\mathrm{CLR}}
=\frac12\left\Vert P(z-u)\right\Vert_2^2.
$$

因为 $P^\top=P$、$P^2=P$，有

$$
\nabla_zL_{\mathrm{CLR}}=P(z-u).
$$

因此 CLR + vanilla gradient 的更新正好是上面的 natural-gradient 更新：

$$
e^+=(I-\eta P)e.
$$

在 quotient space 中，CLR 的 Hessian 是恒等矩阵，不再有 categorical Fisher 的病态谱。

CLR 还可以写成 pairwise objective：

$$
L_{\mathrm{CLR}}
=\frac{1}{4K}\sum_{i,j}
\left[(z_i-z_j)-(u_i-u_j)\right]^2.
$$

这个形式说明 CLR 的本质是匹配所有相对 logits，而不是匹配任意一个绝对 logit。

## 4. 相关论文如何定位 CLR

CLR 不是凭空出现的术语。它位于 direct logit regression、high-temperature distillation、pairwise logit matching 和 score matching 的交集。

### 4.1 Direct logit regression

Ba 和 Caruana 较早直接用 teacher logits 训练 student，发现 logit regression 比 probability regression 和 KL 更有效[^ba14]。Hinton、Vinyals 和 Dean 进一步说明，直接匹配 logits 可以看作高温 distillation 的极限[^hinton15]。

Kim 等人对这一点给出了更精确的推导：若使用带 $\tau^2$ 缩放的 forward KL，则

$$
\lim_{\tau\to\infty}
\nabla_z\left[\tau^2D_{\mathrm{KL}}(q_\tau\Vert p_\tau)\right]
=\frac1K P(z-u).
$$

因此高温 forward KL 与 CLR 共享 output-space vector field，只差整体比例[^kim21]。

### 4.2 Normalized / standardized logit losses

Class-aware Logit Knowledge Distillation 使用 L2-normalized MSE，并加入 class-level correlation loss[^zhang22]。Logit Standardization 则先对 teacher 和 student logits 做 per-example Z-score，再计算 KL[^sun24]。

它们和 CLR 的区别是：

| 方法              | 消除公共平移 | 消除 logit 尺度 | 是否保持 teacher 的 pairwise 尺度 |
| ----------------- | ------------ | --------------- | --------------------------------- |
| Raw DLD           | 否           | 否              | 是，但 gauge 过强                 |
| CLR               | 是           | 否              | 是                                |
| L2-normalized MSE | 间接         | 是              | 否                                |
| Z-score + KL      | 是           | 是              | 否                                |

如果目标是复现 reverse-KL 的 exact output natural-gradient dynamics，CLR 比 Z-score 更直接；如果 teacher/student 的 logit scale 本身不兼容，Z-score 可能是有用的经验方法，但它已经改变了目标。

### 4.3 CSD：最接近 CLR 的现代表述

Concrete Score Distillation（CSD）把离散 score matching 的思想用于 LLM 蒸馏；其基础的 Concrete Score Matching 由 Meng 等人提出[^csm22]。CSD 的 loss 写成加权 pairwise logit-difference regression：

$$
L_{\mathrm{CSD}}(w_1,w_2)
=\frac12\sum_{i,j}w_{1,i}w_{2,j}
\left[(z_i-z_j)-(u_i-u_j)\right]^2.
$$

它的动机是同时解决两个问题：softmax 会压扁长尾 logit 信息，raw DLD 又没有处理 logit shift invariance。论文还把 pairwise 梯度化简为 $O(K)$ 的 weighted-centered logit residual，并用不同的 $w_1,w_2$ 调节 mode-seeking 和 mode-covering 行为[^csd26]。

在 uniform weights 下：

$$
L_{\mathrm{CSD}}(U,U)=\frac{2}{K}L_{\mathrm{CLR}},\qquad
\nabla_zL_{\mathrm{CSD}}(U,U)=\frac{2}{K}P(z-u).
$$

所以，从目标和输出梯度看，CLR 是 CSD 的 uniform-weight 特例。

这意味着 CLR 的合理论文定位不是“首次提出 centered pairwise logit loss”，而是：

> 把 uniform pairwise logit matching 解释为 categorical reverse-KL 的 exact output natural gradient，并将其用于 OPD 的 sampled / top-k 训练。

## 5. Full-vocabulary CLR 的实现

如果 teacher 的完整 logits 可得，推荐直接在 logits 上计算 CLR：

    def clr_loss(student_logits, teacher_logits):
        teacher_logits = teacher_logits.detach()
        delta = student_logits - teacher_logits
        centered = delta - delta.mean(dim=-1, keepdim=True)
        return 0.5 * centered.square().sum(dim=-1).mean()

实现要点：

- teacher 必须 detach；
- 对 vocabulary 求 sum 时不要再额外除以 $K$，否则只是学习率缩放；
- full-vocab 条件下 token sampling variance 为零；
- 若 batch 中有多个 context，剩余的是 context/data variance，而不是 vocabulary sampling variance。

对于直接优化 logits，CLR 的 quotient-space 收敛是确定的几何收缩。对于神经网络，反向传播后实际输出变化还要乘以网络的 tangent kernel，这一点见第 10 节。

## 6. Single-token CLR estimator

### 6.1 从任意 proposal 采样

令

$$
A\sim s,\qquad s_i>0.
$$

记 $e_A$ 为 sampled token 的 one-hot 向量。给定任意不依赖当前采样 $A$ 的 baseline $b$，定义

$$
\widehat h_A
=\frac{\delta_A-b}{s_A}Pe_A.
$$

则

$$
\begin{aligned}
\mathbb E_{A\sim s}[\widehat h_A]
&=\sum_i s_i\frac{\delta_i-b}{s_i}Pe_i \\
&=P\delta.
\end{aligned}
$$

所以一个 token 足以无偏估计完整 CLR gradient，但它是一个 gradient oracle，不是完整 CLR 标量 loss 的无偏估计。

可以通过 surrogate loss 反传：

    def sampled_clr_surrogate(student_logits, token_id, proposal_prob, baseline):
        coefficient = (student_logits[..., token_id] - baseline).detach()
        coefficient = coefficient / proposal_prob.detach()
        centered_logit = (
            student_logits[..., token_id]
            - student_logits.mean(dim=-1)
        )
        return coefficient * centered_logit

importance weight、baseline 和 proposal probability 都应该 stop-gradient。

### 6.2 方差

协方差为

$$
\operatorname{Cov}(\widehat h_A)
=\sum_i\frac{(\delta_i-b)^2}{s_i}
(Pe_i)(Pe_i)^\top
-(P\delta)(P\delta)^\top.
$$

其 trace，即总平方误差方差，为

$$
V_1
=\left(1-\frac1K\right)
\sum_i\frac{(\delta_i-b)^2}{s_i}
-\left\Vert P\delta\right\Vert_2^2.
$$

固定 proposal 时，使二阶矩最小的 baseline 是

$$
b^*(s)
=\frac{\sum_i\delta_i/s_i}{\sum_i1/s_i}.
$$

固定 baseline 时，理论最优 proposal 满足

$$
s_i^*\propto\lvert\delta_i-b\rvert.
$$

这个 proposal 需要知道完整 error，因此实际只能近似。可用的 proposal 包括 teacher top-k、student top-k、历史 error 估计，以及带 uniform tail 的 mixture：

$$
s=(1-\rho)\pi+\rho U,\qquad U_i=\frac1K.
$$

$\rho>0$ 不是装饰：它保证所有 vocabulary token 有非零采样概率。

### 6.3 按 student $p$ 采样时的代价

OPD 通常从 $p$ 采样。此时

$$
V_1
=\left(1-\frac1K\right)
\sum_i\frac{(\delta_i-b)^2}{p_i}
-\left\Vert P\delta\right\Vert_2^2.
$$

当 $p_i$ 很小而 $\delta_i$ 不小时，方差按 $1/p_i$ 增长。这是 naturalization 的基本 trade-off：

- vanilla reverse-KL gradient 有 probability starvation，但 rare token 方差较温和；
- CLR / NGD 恢复 rare-token 的期望更新，但会放大 importance-sampling noise。

若平均 $M$ 个独立 sampled tokens，方差约为 $V_1/M$。

## 7. Sampled reverse-KL 与 CLR 的精确关系

现有 sampled-token OPD 常使用 K2 estimator。令 $A\sim p$，定义

$$
L_A^{\mathrm{K2}}
=\frac12\left(\log p_A-\log q_A\right)^2
=\frac12\delta_A^2.
$$

固定 sampled token、teacher 不反传时：

$$
\widehat g_A^{\mathrm{RKL}}
=\nabla_zL_A^{\mathrm{K2}}
=\delta_A(e_A-p).
$$

取期望：

$$
\mathbb E_{A\sim p}[\widehat g_A^{\mathrm{RKL}}]
=F_p\delta
=\nabla_zD_{\mathrm{KL}}(p\Vert q).
$$

因此 K2 的 scalar value 是有偏的，但在固定 context、当前 student sampling 下，它的 logit gradient 是 reverse-KL gradient 的无偏估计。近期 sampled-token OPD 分析[^opd26] 也从 token-level gradient norm 角度研究了这一 estimator，得到

$$
\left\Vert\widehat g_A^{\mathrm{RKL}}\right\Vert_1
=2\lvert\log q_A-\log p_A\rvert(1-p_A).
$$

对 K2 gradient 做 per-sample categorical Fisher naturalization：

$$
F_p^+(e_A-p)=\frac1{p_A}Pe_A.
$$

于是

$$
F_p^+\widehat g_A^{\mathrm{RKL}}
=\frac{\delta_A}{p_A}Pe_A,
$$

这正是 $s=p$ 的 sampled CLR estimator。

这条关系把三种方法串起来：

$$
\text{K2 sampled reverse-KL}
\xrightarrow{\;F_p^+\;}
\text{sampled CLR}
\xrightarrow{\;\mathbb E\;}
\text{exact output NGD field}.
$$

需要和 Self-Distilled Policy Gradient（SDPG）[^sdpg26] 等论文中的“centered log-ratio policy-gradient advantage”区分：policy-gradient 中常用的是按 student measure 中心化的 baseline $\mathbb E_p[\delta]$；CLR 的 $P\delta$ 是按 vocabulary uniform measure 去除 logit gauge。二者都出现 centered ratio，但不是同一个输出 vector field。

## 8. Two-token pairwise estimator

如果希望同时得到无偏 scalar loss，至少需要两个独立 samples。令 $A,B\sim s$ 独立，则

$$
\widehat L_{\mathrm{pair}}
=\frac{(\delta_A-\delta_B)^2}{4K s_As_B}
$$

满足

$$
\mathbb E[\widehat L_{\mathrm{pair}}]=L_{\mathrm{CLR}}.
$$

其对应 gradient 为

$$
\widehat h_{\mathrm{pair}}
=\frac{\delta_A-\delta_B}{2K s_As_B}(e_A-e_B).
$$

这个 estimator 的优点是：

- scalar loss 本身无偏；
- 每个样本天然 shift-invariant；
- 不需要估计 vocabulary mean；
- gradient 坐标和严格为零。

总方差为

$$
V_{\mathrm{pair}}
=\frac1{2K^2}
\sum_{i,j}
\frac{(\delta_i-\delta_j)^2}{s_is_j}
-\left\Vert P\delta\right\Vert_2^2.
$$

pairwise 并不保证比 single-token 方差更小；它的主要优势是 gauge-invariant 的无偏 scalar objective。

## 9. Top-k 训练：近似目标而不是完整 CLR

### 9.1 Pure head-only CLR

令 $H$ 为 top-k 集合，$\lvert H\rvert=m$。head-only loss 是

$$
L_H
=\frac12\sum_{i\in H}
\left(\delta_i-\bar\delta_H\right)^2.
$$

它在给定 $H$ 后是确定性的，因此没有 token sampling variance，但相对于完整 CLR 有 bias。它只约束：

- head 内部的相对 logits；
- 不约束 head 与 tail 的整体质量比例；
- 不约束 tail 内部的相对结构。

所以 head-only 可能出现“top-k 内部已经匹配，但 student 的 head mass 仍然错误”。

### 9.2 Head + OTHER

如果知道 tail 总质量：

$$
p_T=1-\sum_{i\in H}p_i,\qquad
q_T=1-\sum_{i\in H}q_i,
$$

可以把 tail 合并成一个 OTHER 类，形成 $k+1$ 类 CLR：

$$
\tilde\delta_i=\log p_i-\log q_i,\quad i\in H,
$$

$$
\tilde\delta_T=\log p_T-\log q_T.
$$

然后在这 $k+1$ 个 coarse classes 上做中心化回归。它保留 head 内部关系和 head-tail 总 mass，但放弃 tail 内部结构。

如果 top-k API 返回的是重新归一化后的 top-k probabilities，就不能把缺失部分直接当成 $1-\sum_{i\in H}p_i$；必须知道原始 full-vocab mass 或额外估计 tail mass。

### 9.3 Head 精确、tail 随机

利用 pairwise 表示：

$$
L_{\mathrm{CLR}}=L_{HH}+L_{HT}+L_{TT}.
$$

设 $m=\lvert H\rvert$、$n=\lvert T\rvert$。$L_{HH}$ 精确计算；若 $I\sim U(H)$、$J\sim U(T)$，则

$$
\widehat L_{HT}
=\frac{mn}{2K}(\delta_I-\delta_J)^2
$$

无偏估计 head-tail 项。若 $J_1,J_2\sim U(T)$，则

$$
\widehat L_{TT}
=\frac{n^2}{4K}(\delta_{J_1}-\delta_{J_2})^2
$$

无偏估计 tail-tail 项。

这是一种更接近完整 CLR 的 top-k 方案：head 用确定性信号，tail 用有限预算探索。

### 9.4 与已有 top-k 论文的关系

BiLD 使用 teacher/student top-k logits 的 pairwise differences，重点是过滤 long-tail noise 和保留 ranking；它是有意设计的 truncated objective，不是完整 CLR 的无偏估计[^bild25]。

Pre-training Distillation 的设计空间研究[^peng25] 和 Sparse Logit Sampling[^anshumann25] 都指出，直接缓存 teacher top-k 概率会产生 bias 或 calibration 问题；从 teacher distribution 随机采样可以在期望上保留 forward-KL gradient。但这些方法解决的是 forward-KL 的稀疏化，不会自动解决 CLR estimator 的 $1/s_i$ 方差。

Tail-aware Distillation 将 teacher top-k 与 tail KL 分开，并重新加权 tail，使 top-k 不再吞掉全部学习信号[^tad26]。它与这里的 head + OTHER 和 head + tail sampling 是同一类工程思想：如果只看到 head，必须显式处理 tail mass 或 tail structure。

## 10. 神经网络参数空间：CLR 不等于完整 parameter NGD

令

$$
z=z_\theta(c),\qquad
J=\frac{\partial z}{\partial\theta}.
$$

output-space Fisher 是 $F_z$，parameter-space Fisher 近似为

$$
F_\theta=J^\top F_zJ.
$$

CLR 对参数的普通梯度是

$$
g_\theta^{\mathrm{CLR}}=J^\top P(z-u).
$$

网络输出的变化则近似为

$$
\Delta z
\approx-\alpha JJ^\top P(z-u).
$$

所以只有当 $JJ^\top$ 在任务 tangent space 上近似各向同性时，参数 VGD 才会近似理想的 output-space CLR dynamics。

完整 parameter NGD 要求解

$$
(J^\top F_zJ+\lambda I)\Delta\theta
=-g_\theta.
$$

这也是 K-FAC、trust-region 和 conjugate-gradient 方法的基本方向。Amari 奠定了 natural gradient 的信息几何基础，Kakade 将其用于 policy gradient，K-FAC 则用 layerwise Kronecker factorization 近似大规模 neural-network Fisher[^amari98][^kakade01][^pascanu13][^kfac15][^martens20]。

对最后一层 $z=Wh$，常见近似是

$$
F_W
\approx
\mathbb E[F_z]\otimes\mathbb E[hh^\top].
$$

这给出一个实际可行的两边 whitening：一边处理 vocabulary/output curvature，另一边处理 hidden-feature covariance。

## 11. Adam、对角 Fisher 与低成本修复

Adam 的更新是

$$
\Delta\theta_t
\propto
\frac{m_t}{\sqrt{v_t}+\epsilon}.
$$

它是 coordinate-wise 的历史二阶矩归一化，不是一般意义上的

$$
F_\theta^{-1}g_\theta.
$$

它不处理 token 间的 Fisher off-diagonal coupling，也不保证 logit quotient 中每个 pairwise error 等速收缩。

如果已经有 output logits 的 reverse-KL gradient $g_z=F_p\delta$，则：

- 最便宜的 exact output correction 是直接使用 $P\delta$；
- 若只能从 sampled K2 gradient 开始，可使用 $F_p^+$ 的解析作用；
- 对 generic output gradient，不能简单把每个坐标除以 $p_i(1-p_i)$ 就声称得到了 exact NGD；
- 在这个特殊 reverse-KL 结构中，$\operatorname{diag}(p)$ 加上正确的中心化关系比 generic Fisher diagonal 更自然；
- rare token 方差过大时使用 damping、proposal mixture、gradient clipping 或增大 sample batch，但 clipping 会引入 bias。

一种实际的 output-space sampled CLR 更新是：

$$
\widehat h_A
=\frac{\delta_A-b}{p_A}Pe_A,
$$

然后把 $\widehat h_A$ 作为 logits 的外部 gradient 传回网络。它实现的是 output-space naturalization；它不等于完整 parameter-space NGD，但通常比直接使用 $\delta_A(e_A-p)$ 更接近理想的分布空间方向。

## 12. 推荐的训练配方

### 12.1 Full-vocabulary baseline

第一版应实现：

$$
L_{\mathrm{CLR}}
=\frac12\left\Vert
(z-u)-\operatorname{mean}(z-u)\mathbf1
\right\Vert_2^2.
$$

使用普通 SGD 作为最干净的 VGD baseline；同时记录 Adam 版本，但不要把 Adam 的结果解释成 natural gradient。

### 12.2 Sampled-token 版本

建议至少比较三种：

1. K2：$\delta_A(e_A-p)$，估计普通 reverse-KL gradient；
2. Fisher-naturalized K2：$\delta_A Pe_A/p_A$，估计 CLR gradient；
3. proposal CLR：$(\delta_A-b)Pe_A/s_A$，使用带 uniform tail 的 proposal。

每个版本都记录：sample count、importance-weight 分位数、gradient norm、covariance trace 和 rare-token 覆盖率。

### 12.3 Top-k 版本

建议顺序是：

1. head-only CLR，作为最简单 biased baseline；
2. head + OTHER，测试 tail mass 是否是主要问题；
3. head 精确 + tail pair sampling，测试完整 CLR 的 tail structure 是否重要；
4. CSD(U,S) / CSD(T,S)，测试加权 pairwise objective 能否在不显式采样 tail 的情况下改善训练。

### 12.4 最小实验矩阵

固定一个 context 和一个小 vocabulary，直接画出：

| 方法             | 预期 logit dynamics        | 预期方差                              |
| ---------------- | -------------------------- | ------------------------------------- |
| Reverse-KL + VGD | $I-\alpha F_p$，多时间尺度 | sampled 时较温和                      |
| Reverse-KL + NGD | $I-\eta P$，等速收缩       | exact full-vocab 无 sampling variance |
| Full CLR + VGD   | $I-\eta P$，等价 NGD       | 0                                     |
| K2 sampled       | 期望为 $F_p\delta$         | rare-token 不被逆概率放大             |
| Sampled CLR      | 期望为 $P\delta$           | 按 $1/s_i$ 放大                       |
| Pure top-k CLR   | head 内部收缩              | 条件于 top-k 为 0，但有 tail bias     |

至少报告以下五个量：

1. quotient logit error $\Vert P(z-u)\Vert_2$；
2. reverse KL 和 forward KL；
3. head mass / tail mass；
4. sampled gradient covariance trace；
5. 每个 optimizer step 的实际输出 logit change。

## 13. 最终结论

整个问题可以压缩成一条链：

$$
\text{reverse KL}
\longrightarrow
\text{categorical Fisher gradient }F_p\delta
\longrightarrow
\text{Fisher inverse}
\longrightarrow
\text{centered logit error }P\delta.
$$

普通 VGD 停在第二步，所以它会受到 probability-dependent curvature 的影响；CLR 直接写出了第四步，因此在 output quotient space 中复现了 ideal natural-gradient dynamics。

已有文献分别覆盖了这条链上的不同片段：direct logit regression、high-temperature KD、CSD 的 pairwise shift-invariant loss、reverse-KL OPD、sampled-token gradient、sparse/top-k distillation，以及 K-FAC 等 parameter-space natural-gradient approximation。真正值得继续推进的组合是：

> 用 CLR / uniform CSD 作为 reverse-KL 的 output-space naturalized target，再系统处理 sampled-token inverse-Fisher 的方差、top-k tail bias 和网络参数空间的 Jacobian conditioning。

这比单纯把 CLR 当作另一个 MSE loss 更准确，也更容易形成可验证的理论和实验贡献。

### 参考文献

[^ba14]: Ba, Lei Jimmy, and Rich Caruana. Do Deep Nets Really Need to be Deep? NeurIPS 2014. [arXiv:1312.6184](https://arxiv.org/abs/1312.6184)

[^hinton15]: Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. NeurIPS Deep Learning Workshop 2014. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

[^kim21]: Kim, Taehyeon, Jaehoon Oh, Nak Yil Kim, Sangwook Cho, and Se-Young Yun. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation. IJCAI 2021. [arXiv:2105.08919](https://arxiv.org/abs/2105.08919)

[^zhang22]: Zhang, Shuoxi, and Hanpeng Liu. Class-aware Information for Logit-based Knowledge Distillation. arXiv 2022. [arXiv:2211.14773](https://arxiv.org/abs/2211.14773)

[^sun24]: Sun, Shangquan, Wenqi Ren, Jingzhi Li, Rui Wang, and Xiaochun Cao. Logit Standardization in Knowledge Distillation. CVPR 2024. [arXiv:2403.01427](https://arxiv.org/abs/2403.01427)

[^csm22]: Meng, Chenlin, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete Score Matching: Generalized Score Matching for Discrete Data. NeurIPS 2022. [arXiv:2211.00802](https://arxiv.org/abs/2211.00802)

[^csd26]: Kim, Yeongmin, Donghyeok Shin, Mina Kang, Byeonghu Na, and Il-Chul Moon. Distillation of Large Language Models via Concrete Score Matching. ICLR 2026. [arXiv:2509.25837](https://arxiv.org/abs/2509.25837)

[^minillm24]: Gu, Yuxian, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: On-Policy Distillation of Large Language Models. ICLR 2024. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)

[^gkd24]: Agarwal, Rishabh, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

[^opd26]: Liu et al. A Token-Level Analysis of Sampled-Token Reverse-KL On-Policy Distillation. arXiv 2026. [arXiv:2608.25643](https://arxiv.org/abs/2608.25643)

[^sdpg26]: Liu, Yifeng, Shiyuan Zhang, Yifan Zhang, and Quanquan Gu. Self-Distilled Policy Gradient. arXiv 2026. [arXiv:2606.04036](https://arxiv.org/abs/2606.04036)

[^bild25]: Li, Mingsheng, Fei Zhou, and Xin Song. BiLD: Bi-directional Logits Difference Loss for Large Language Model Distillation. COLING 2025. [arXiv:2406.13555](https://arxiv.org/abs/2406.13555)

[^peng25]: Peng, Hao, Xin Lv, Yushi Bai, Zijun Yao, Jiajie Zhang, Lei Hou, and Juanzi Li. Pre-training Distillation for Large Language Models: A Design Space Exploration. ACL 2025. [arXiv:2410.16215](https://arxiv.org/abs/2410.16215)

[^anshumann25]: Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, and Joohyung Lee. Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs. ACL 2025. [arXiv:2503.16870](https://arxiv.org/abs/2503.16870)

[^tad26]: Dasgupta, Sayantani, Trevor Cohn, and Timothy Baldwin. Don’t Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation. arXiv 2026. [arXiv:2602.20816](https://arxiv.org/abs/2602.20816)

[^amari98]: Amari, Shun-ichi. Natural Gradient Works Efficiently in Learning. Neural Computation 1998. [DOI](https://doi.org/10.1162/089976698300017746)

[^kakade01]: Kakade, Sham M. A Natural Policy Gradient. NeurIPS 2001. [论文页面](https://papers.neurips.cc/paper/2073-a-natural-policy-gradient)

[^pascanu13]: Pascanu, Razvan, and Yoshua Bengio. Revisiting Natural Gradient for Deep Networks. ICLR 2013. [arXiv:1301.3584](https://arxiv.org/abs/1301.3584)

[^kfac15]: Martens, James, and Roger Grosse. Optimizing Neural Networks with Kronecker-factored Approximate Curvature. ICML 2015. [arXiv:1503.05671](https://arxiv.org/abs/1503.05671)

[^martens20]: Martens, James. New Insights and Perspectives on the Natural Gradient Method. JMLR 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)
