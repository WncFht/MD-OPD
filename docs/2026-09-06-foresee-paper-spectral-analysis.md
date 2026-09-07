# 《Learning to Foresee》中的 OPD 现象：谱吸收视角

本文分析 Cai et al. 的论文 _Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation_（arXiv:2605.11739）[^cai26]。重点不是复述论文，而是把它的实验现象放进我们已有的“谱吸收”框架中，并说明 EffOPD 在这一框架下实际做了什么。

## 1. 论文主张的最小摘要

论文把 OPD 相对于 RL 的效率概括为两种“foresight”：

1. **Functional Redundancy Avoidance**：OPD 将更新集中到对推理更敏感的模块，尤其是中间层 MLP；RL 在底部和顶部等低边际收益模块上产生更大的冗余更新。
2. **Early Low-Rank Lock-in**：OPD 的更新矩阵更低秩，主奇异子空间在训练早期就接近最终子空间，后续训练主要增加这些方向上的幅度。

论文给出的代表性几何指标是：在 8B 模型上，OPD 的 spectral/Frobenius ratio 为 36.8%（RL 为 32.7%），effective rank 为 2341（RL 为 2754），Top-1% subspace norm ratio 为 94.7%（RL 为 88.5%）。在中间 checkpoint 上保留更新方向、只把各模块的范数缩放到最终值，约 10% 训练进度的 OPD checkpoint 可以恢复约 80% 的最终推理性能。论文据此提出 EffOPD：在指数间隔 checkpoint 上估计局部位移方向，并沿该方向逐级外推；每次外推用 50 个训练样本做轻量验证，性能不再提升时立即停止。论文报告平均约 3 倍训练加速[^cai26]。

这里需要先区分两件事：论文直接测量的是**参数更新矩阵的奇异子空间和任务性能**；我们谱吸收模型讨论的是**目标曲率特征方向上的误差弛豫**。两者相关，但并不自动相同。下面的推导说明它们在何种条件下可以对应。

## 2. 论文附录的局部二次模型

对一个上下文 $c$，令学生和教师 logits 分别为 $z_\theta(c)$、$z^*(c)$，定义

$$
r_c=z^*(c)-z_0(c),\qquad
z_\theta(c)-z_0(c)\simeq J_c\Delta\theta .
$$

在 logits 接近的区域，reverse-KL 可用 Fisher 二次化：

$$
\operatorname{KL}(p_\theta\|p^*)
\simeq \frac12(J_c\Delta\theta-r_c)^\top F_c(J_c\Delta\theta-r_c).
$$

对上下文取平均，得到

$$
\mathcal L(\Delta\theta)
\simeq \frac12\Delta\theta^\top A\Delta\theta-b^\top\Delta\theta+\mathrm{const},
$$

其中

$$
A=\mathbb E_c[J_c^\top F_cJ_c],\qquad
b=\mathbb E_c[J_c^\top F_cr_c].
$$

这正是我们此前使用的 Fisher/NTK 局部几何：$A$ 是参数空间中的有效曲率，$b$ 是教师残差经过输出 Fisher 和 Jacobian 投影后形成的驱动力。固定步长梯度下降为

$$
\Delta\theta_{s+1}=(I-\eta A)\Delta\theta_s+\eta b.
$$

从 $\Delta\theta_0=0$ 出发，其闭式解为

$$
\Delta\theta_s=[I-(I-\eta A)^s]A^\dagger b.
$$

若 $A u_i=\lambda_i u_i$、$b_i=\langle b,u_i\rangle$，则

$$
\Delta\theta_s
=\sum_{i:\lambda_i>0}
\frac{1-(1-\eta\lambda_i)^s}{\lambda_i},b_i u_i.
$$

这组公式是理解论文现象的关键：每个曲率方向都有一个自己的增长时间 $1/(\eta\lambda_i)$，但最终幅度由 $b_i/\lambda_i$ 决定。

## 3. “早期 low-rank lock-in”如何从谱模型出现

### 3.1 驱动力集中，而不是参数本身神秘地变低秩

论文给出的充分条件是：存在一个低维主子空间 $U_k=\operatorname{span}(u_1,\ldots,u_k)$，使得

$$
\|P_{U_k^\perp}b\|\le\varepsilon\|b\|,\qquad \varepsilon\ll1,
$$

并且存在谱隙 $\lambda_k\gg\lambda_{k+1}$。在这个条件下，训练早期的更新主要来自 $U_k$；尾部的范数受

$$
\left\|[I-(I-\eta A)^s]A^\dagger b_\perp\right\|
\le
\max_{i>k}\frac{|1-(1-\eta\lambda_i)^s|}{\lambda_i}
\,\|b_\perp\|
$$

控制。于是“lock-in”不是说网络从一开始就只有 $k$ 个自由度，而是说：**教师残差真正驱动的方向已经集中，低曲率尾部又需要更长时间才能显现。**

这与我们在吸收率模型中的描述完全一致。误差在第 $i$ 个方向上的剩余量按 $e^{-\eta\lambda_i t}$ 衰减，高曲率方向先被消耗；如果初始振幅 $a_i$ 主要落在少数方向，剩余更新和剩余误差都会表现出低维结构。

### 3.2 为什么范数缩放能恢复早期性能

设第 $i$ 个方向的最终更新为

$$
x_i^*=\frac{b_i}{\lambda_i},
$$

则第 $s$ 步的比例为

$$
q_i(s)=\frac{x_i(s)}{x_i^*}=1-(1-\eta\lambda_i)^s.
$$

如果主方向的 $q_i(s)$ 已经形成而只是整体偏小，那么把模块范数乘到最终值，相当于把这些方向的幅度从 $q_i(s)x_i^*$ 直接补到接近 $x_i^*$，而不改变方向。论文“10% 进度恢复约 80% 性能”的结果因此支持如下解释：

> 早期瓶颈主要是沿正确方向走得不够远，而不是还没有找到正确方向。

但这不是任意情况下都成立。若尾部方向的 $b_i$ 不可忽略、模块间耦合很强，或模型已经离开局部二次区域，单纯范数缩放会改变不同方向的相对比例，可能导致过冲。论文的干预实验说明在其测试设置中方向误差较小，但没有证明这一性质对所有任务都成立。

### 3.3 Functional Redundancy Avoidance 的谱解释

把参数按模块分块，记

$$
b_m=\mathbb E_c[J_{c,m}^\top F_cr_c],\qquad
A_{mn}=\mathbb E_c[J_{c,m}^\top F_cJ_{c,n}].
$$

若模块间耦合不强，则局部最优更新近似为

$$
\Delta\theta_m^*\simeq A_{mm}^{\dagger}b_m.
$$

因此，当某个模块对教师残差的投影 $b_m$ 很小，它的最优更新也很小。论文观察到 embedding、底部和顶部模块的干预收益较低，同时 RL 在这些模块上积累了更大的范数；这可以解释为 RL 的高方差驱动力在低 $b_m$ 区域仍然产生了随机游走，而 OPD 的 dense residual 使更新更接近 $b_m$ 的真实结构。

这里“冗余”是相对于目标性能而言的，不是说这些参数变化在欧氏范数意义上不存在。换言之，论文的 module allocation 结果可以看作对驱动力 $b$ 的分块测量，而不是新的优化目标。

## 4. OPD 与 RL 的差异：不仅是监督更密

论文把 OPD 的初始驱动力近似为

$$
\hat g_{\mathrm{OPD}}=J_c^\top F_cr_c,
$$

而 RL 的单 token 梯度近似为

$$
\hat g_{\mathrm{RL}}=A\,J_c^\top(e_y-p_0(c)).
$$

两者的差异可以在我们的谱语言中重述为：

- OPD 的 $r_c$ 是教师和学生之间的方向性残差，经过 $F_c$ 过滤后直接指向需要修正的输出模式；
- RL 的标量 advantage 乘上随机的 $e_y-p_0$，在稀疏奖励下方差更大，因而更容易激活与最终 $b$ 不一致的参数方向。

这会同时造成两种后果。第一，OPD 的有效驱动力 $b$ 更集中，早期就建立主子空间；第二，RL 的噪声在平坦方向上积累，形成论文所说的低边际收益更新。它与我们噪声底推导的关系是：平坦模式的收缩率 $\eta\lambda_i$ 小，但稳态位移方差约为

$$
\mathbb E[x_i^2]_\infty\simeq\frac{\eta\sigma_i^2}{2\lambda_iB}.
$$

因此，“RL 更新范数更大但性能更低”并不矛盾：一部分范数来自高 $\sigma_i$、低 $\lambda_i$ 的噪声驱动，而不是来自更大的任务相关信号。

## 5. EffOPD 在谱模型下做了什么

### 5.1 算法步骤

论文在 $t=2^n$ 的 checkpoint 触发一次外推。定义相邻指数 checkpoint 的位移

$$
\Delta_n=W_{2^n}-W_{2^{n-1}}.
$$

从当前参数 $W_{2^n}$ 出发，依次考察

$$
\widetilde W_{n,k}=W_{2^n}+2^k\Delta_n,
\qquad k=1,\ldots,5.
$$

每个候选用 50 个训练样本评估；只要验证分数不下降就接受，第一次失败便停止。若 $k=1$ 就失败，则退化为 vanilla OPD[^cai26]。

### 5.2 对单一谱模式的推导

为看清外推的含义，考虑一个固定曲率模式

$$
x_t=x^*(1-q^t),\qquad q=1-\eta\lambda.
$$

指数区间位移为

$$
\Delta_n=x_{2^n}-x_{2^{n-1}}
=x^*\left(q^{2^{n-1}}-q^{2^n}\right).
$$

它是该模式在最近一段时间的“局部速度”乘以区间长度。EffOPD 使用 $2^k\Delta_n$，本质上是在假设接下来的轨迹仍沿同一方向、并把局部速度持续一段时间。对尚未饱和的模式，这会比再走一个普通 OPD step 更快地增加 $x_t$；对已经接近饱和的模式，外推则容易过冲。

多模式情形下，若主方向的 $q_i$ 和 $b_i$ 结构相近，位移向量

$$
\Delta_n=\sum_i\left[x_i(2^n)-x_i(2^{n-1})\right]u_i
$$

会稳定在同一个主子空间内，外推主要增加其幅度；这正是论文“方向早锁定、后期主要长幅度”的条件。反过来，如果方向还在旋转，$\Delta_n$ 只是两个 checkpoint 的差向量，外推会把方向误差也一并放大。

### 5.3 为什么需要验证集

谱模型给出的安全条件是理想化的：固定 $A$、固定 $b$、确定性梯度和局部二次目标。真实 OPD 还包含 Adam 预条件、on-policy 状态变化、随机 rollout 和模型非线性。于是 EffOPD 的 50 样本验证承担了一个轻量 trust-region / line-search 的作用：它不估计完整梯度，只检查“沿当前位移方向再走几步是否仍改善任务分数”。

论文关于学习率的消融也符合这一解释：更大的 vanilla OPD 学习率虽然早期更快，但会产生振荡；EffOPD 通过逐级候选和失败即停，截断了过大的外推。验证集难度不敏感，则说明它主要提供方向性排序，而不是高精度的损失估计。

不过，验证接受并不等于真实 OPD 目标一定下降，原因有三：样本只有 50 个，验证噪声可能误接受；验证分布与后续 on-policy 状态可能不同；外推后的状态分布可能改变，导致原有方向不再有效。因此 EffOPD 更准确的定位是**带任务反馈的方向外推器**，而不是谱模型的无条件闭式加速。

## 6. 与我们吸收率理论的对应关系

可以把两套叙述对齐如下：

| 《Learning to Foresee》的术语 | 谱吸收模型中的对应物                         |
| ----------------------------- | -------------------------------------------- |
| 主奇异子空间早期稳定          | 驱动力 $b$ 集中于少数 $A$ 特征方向           |
| 后期主要增加 update magnitude | 主方向的 $q_i(s)$ 继续增长，方向权重变化较小 |
| RL 的 redundant updates       | 高噪声、低曲率方向的随机位移                 |
| 模块边际收益低                | 模块驱动力 $b_m$ 小或位于谱尾                |
| 范数缩放恢复早期性能          | 保持方向不变，直接增大 $q_i(s)x_i^*$         |
| EffOPD 外推                   | 沿局部位移近似的主谱方向增加有效时间         |
| 外推过大导致振荡              | 超出局部二次区或进入时变谱/噪声底区域        |

这里最重要的区别是：论文的 SVD 是对每个参数矩阵的欧氏更新 $\Delta W$ 做分解，而我们的 $A$-谱是 Fisher/NTK 度量下的曲率分解。只有当参数矩阵的主奇异方向与 $A$ 的主特征方向大致对齐时，“low-rank update”才等价于“少数快速吸收模式”。如果参数化存在强烈的各向异性，欧氏 SVD 可能高估或低估真正的自然梯度方向集中度。

## 7. 对论文实验现象的逐条判断

### 7.1 OPD 在相同 update norm 下性能更高

这与“有效驱动力集中”一致。若把最终更新写成 $\Delta\theta^*=A^\dagger b$，固定 $\|\Delta\theta\|$ 后，性能取决于更新与 $b$ 以及目标曲率的匹配，而不只取决于范数。OPD 若有更高的 $b$-对齐度，就能在同样范数下获得更大的 KL 下降和推理收益。RL 的额外范数可以落在低 $b$、高噪声方向。

### 7.2 中间层 MLP 更敏感，底部/顶部更新更冗余

在模块块矩阵中，这表示中间 MLP 的 $b_m$ 或有效耦合更大，底部和顶部模块的 $b_m$ 较小。论文观察到 OPD 与 RL 的干预敏感性分布相似，但 RL 的低敏感模块范数更大，说明两者利用的是相近的功能通路，差别主要在驱动力和噪声分配，而不是完全不同的知识存储位置。

### 7.3 OPD 的主子空间更早接近最终子空间

这对应谱模型中的“驱动力先确定、幅度后增长”。当 $b$ 已集中且谱隙存在时，主方向从第一步就存在；随时间变化的主要是 $1-(1-\eta\lambda_i)^s$，而不是方向本身。RL 的高方差梯度会让有限样本的主奇异向量产生更大旋转，因此早期到最终子空间的 cosine similarity 较低。

### 7.4 EffOPD 约 3 倍加速

在理想固定谱上，普通梯度下降需要许多步把 $q_i(s)$ 推近 1；EffOPD 用位移外推直接增加有效幅度，因此可以减少“沿已知方向走路”的迭代次数。它不能消除低曲率模式的统计噪声底，也不能凭空发现 $b_i\approx0$ 的方向，所以加速主要应发生在“方向已经稳定但幅度仍不足”的阶段，而不是最终极限附近。

论文报告 EffOPD 有时达到比 vanilla OPD 更高的性能上限，并将其归因于 vanilla OPD 长时间训练可能出现 over-optimization 或 semantic drift。按我们的模型，这意味着外推不仅节省了时间，还可能减少在时变状态分布和噪声底附近继续更新的步数；但这属于经验现象，不是固定二次模型的结论。

## 8. 结论与可验证边界

这篇论文为我们的谱吸收理论补上了一个重要的“方向形成”层面：

$$
\text{教师残差 }r
\xrightarrow{\;J^\top F\;}
\text{低维驱动力 }b
\xrightarrow{\;A\text{ 的谱响应}\;}
\text{早期方向锁定、后期幅度吸收}.
$$

在这个链条中，谱吸收模型解释吸收率为何下降；论文解释为什么 OPD 的有效谱质量可能从一开始就集中在少数方向，以及为什么 RL 更容易把预算花在低边际收益方向。EffOPD 则利用这一结构，沿当前局部位移估计的主方向做带验证的外推。

最需要保留的边界有三条：

- 参数矩阵的欧氏低秩不等于 Fisher/NTK 意义下的低秩，二者需要额外对齐证据。
- 局部二次模型要求固定状态分布和近似固定曲率；Adam、on-policy 漂移和 progressive sharpening 都会破坏这一条件。
- 验证集接受只说明候选在有限样本上更好，不保证真实 KL、长期吸收率或最终泛化一定更好。

因此，对这篇论文最简洁而准确的评价是：**它把 OPD 的效率拆成“更早找到有效谱子空间”和“在该子空间里更少浪费更新”两部分；我们的谱吸收理论则给出这两部分如何由 Fisher 曲率、教师残差投影和噪声底共同产生的统一解释。**

### 参考文献

[^cai26]: Y. Cai, D. Cao, L. Lin, et al. _Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation_. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^fu26]: Z. Fu, B. He, Y. Zuo, et al. _Rethinking On-Policy Distillation of Large Language Models II: One Training Example_. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^amari98]: S. Amari. _Natural Gradient Works Efficiently in Learning_. Neural Computation 10(2):251--276, 1998.

[^martens14]: J. Martens. _New Insights and Perspectives on the Natural Gradient Method_. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^bordelon20]: A. Bordelon, A. Canatar, C. Pehlevan. _Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks_. ICML 2020. [arXiv:2002.02561](https://arxiv.org/abs/2002.02561)

[^mandt17]: S. Mandt, M. Hoffman, D. Blei. _Stochastic Gradient Descent as Approximate Bayesian Inference_. JMLR 18(134):1--35, 2017. [arXiv:1704.04289](https://arxiv.org/abs/1704.04289)

[^kingma15]: D. Kingma, J. Ba. _Adam: A Method for Stochastic Optimization_. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

[^cohen21]: J. Cohen, A. Kaur, Y. Li, J. Z. Kolter, A. Talwalkar. _Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability_. ICLR 2021. [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
