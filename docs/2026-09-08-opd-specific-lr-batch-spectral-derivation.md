# OPD 场景下的学习率、Batch Size 与谱吸收推导

**日期**：2026-09-08

**定位**：把学习率、Batch Size、EMA、Adam 与 Hessian 的局部结论映射到本项目的 OPD 训练路径，形成可测量、可证伪的 OPD-specific 推导。

本文以当前冻结态基线为准：**token_reward_direct**、**token-mean**、**opd_noratio**、**MINI_BATCH_SIZE=64**、**N_RESPONSES=1**、**LOG_PROB_TOP_K=0**、**METRIC_TOP_K=16**。重点区分三件容易混淆的事：完整序列 OPD 梯度、当前代码使用的 immediate-token estimator，以及 **LOG_PROB_TOP_K** 打开的候选 token policy-loss 路径。

代码锚点：

- 冻结态启动器：[scripts/run_frozen_state_opd.sh](../scripts/run_frozen_state_opd.sh)；
- 公共训练参数与 top-$k$ 开关：[recipe/\_common.sh](../recipe/_common.sh)；
- OPD advantage 与 opd_noratio：[verl/verl/trainer/ppo/opd_core_algos.py](../verl/verl/trainer/ppo/opd_core_algos.py)；
- teacher reward、top-$k$ 语义与 frozen 校验：[verl/verl/trainer/ppo/opd_trainer_mixin.py](../verl/verl/trainer/ppo/opd_trainer_mixin.py)；
- frozen rollout 的约束：[verl/verl/trainer/ppo/ray_trainer.py](../verl/verl/trainer/ppo/ray_trainer.py)。

## 摘要

固定状态、教师固定、局部可表示且梯度裁剪不主导时，OPD 的局部目标可以写成

$$
J_{\mathcal S}(\theta)
=\mathbb E_{s\sim\mu_{\mathcal S}}
\operatorname{KL}\left(q_\theta(\cdot\mid s)\Vert p_T(\cdot\mid s)\right).
$$

在教师分布落入学生模型族的匹配点，OPD Hessian 是 student token 分布的 Fisher/Gauss--Newton 矩阵。普通一阶更新沿这个谱逐方向收缩，高曲率方向先被吸收，尾部由低曲率方向和噪声底控制。

OPD 的 Batch Size 不能只写成 rollout 数。相邻 response token、共享 prefix、后缀 reward，以及同一 prompt 的多条 response 都会引入相关性。令 $R$ 为总 rollout 数，$L_{\mathrm{eff}}$ 为消除序列内自相关后的有效长度，$D_{\mathrm{cluster}}$ 为 prompt cluster 设计效应，则

$$
\boxed{
B_{\mathrm{OPD}}
\approx
\frac{R L_{\mathrm{eff}}}{D_{\mathrm{cluster}}}
}.
$$

本文的主要结论是：

1. 当前 **token_reward_direct** 是逐 token 的 immediate-token estimator，不是完整序列后缀累计梯度；
2. 固定状态的 OPD 局部曲率为 $F_{\mathrm{OPD}}$，但只在分布匹配或局部小失配时可把失配残差忽略；
3. 在固定谱和小步长近似下，L1 advantage 距离的吸收率约为 $\eta\langle\lambda\rangle$，局部 KL 距离约为 $2\eta\langle\lambda\rangle_{\mathrm{KL}}$；
4. 增大有效 token 数、rollout 数或减少 cluster 相关性主要降低噪声底，不必改变冻结态的固定谱形状；
5. $\beta_1=0.9$ 在独立跨步噪声近似下给一阶 EMA 约 $19$ 倍的有效 Batch 放大，但 $\beta_2=0.999$ 的时间常数约为 $1000$ 个 optimizer step，300 step 的 run 仍处在二阶状态瞬态内；
6. **LOG_PROB_TOP_K** 改变训练 reward 与 policy loss，**METRIC_TOP_K** 只改变观测信号；
7. frozen 与 on-policy 的速率差可用状态分布漂移项诊断，而不应直接称为 Fisher 谱变化。

这些是有明确前提的局部推导和可证伪预测，不是对完整非线性 OPD 训练的无条件收敛定理。

## 1. 当前 OPD 基线

### 1.1 实际配置

下表对应 [scripts/run_frozen_state_opd.sh](../scripts/run_frozen_state_opd.sh) 通过 [recipe/opd.sh](../recipe/opd.sh) 传入的冻结态基线。它比通用 recipe/opd.sh 更具体：通用 launcher 在非 top-$k$ 时默认 vanilla，冻结态脚本显式覆盖为 opd_noratio。

| 配置                         |          冻结态基线 | 在本文中的含义                             |
| ---------------------------- | ------------------: | ------------------------------------------ |
| data.train_batch_size        |                  64 | 每个 optimizer step 的 prompt/rollout 行数 |
| actor.ppo_mini_batch_size    |                  64 | 一次更新覆盖整个训练 batch                 |
| actor.ppo_epochs             |                   1 | 不重复消费同一 batch                       |
| rollout.n                    |                   1 | 每个 prompt 只生成一条 response            |
| algorithm.adv_estimator      | token_reward_direct | 每个有效 token 直接使用 teacher reward     |
| actor.loss_agg_mode          |          token-mean | 在有效 response token 上做全局平均         |
| actor.policy_loss.loss_mode  |         opd_noratio | 不使用 PPO ratio，也不使用 rollout IS      |
| optimizer                    |          FSDP AdamW | 一阶与二阶 EMA 共同决定更新                |
| learning rate                |                1e-6 | 冻结态谱时间尺度的基线                     |
| betas                        |        (0.9, 0.999) | 一阶降噪与二阶状态记忆                     |
| clip_grad                    |                 1.0 | 可能改变大梯度尾部与噪声统计               |
| LOG_PROB_TOP_K               |                   0 | 不把 top-$k$ 候选维度放进训练 reward       |
| METRIC_TOP_K                 |                  16 | 只记录 top-$k$ agreement/mass 等指标       |
| trainer.total_training_steps |                 300 | frozen run 的默认窗口                      |

token-mean 很重要。若改成 sequence-mean 或 token-sum，长 response 的权重会变化，$B_{\mathrm{OPD}}$ 与梯度协方差的解释也要相应重写。这里的 64 是 batch 中的 rollout 行数，不是 64 个独立 token，也不自动等价于 64 个独立梯度样本。

配置与实现的具体来源是 [recipe/\_common.sh](../recipe/_common.sh)、[verl/verl/trainer/config/distillation/distillation.yaml](../verl/verl/trainer/config/distillation/distillation.yaml) 与 [verl/verl/workers/config/optimizer.py](../verl/verl/workers/config/optimizer.py)。本文默认把这些当前代码事实与理论假设分开书写：配置是事实，谱和噪声公式是带条件的模型。

### 1.2 token_reward_direct 的实际梯度

固定一个状态 $s$，记

$$
p_s(y)=\pi_T(y\mid s),
\qquad
q_\theta(y)=\pi_\theta(y\mid s),
$$

并定义

$$
u_\theta(y)=\nabla_\theta\log q_\theta(y),
\qquad
\Delta_\theta(y)=\log q_\theta(y)-\log p_s(y).
$$

单状态 reverse-KL 为

$$
J_s(\theta)
=\sum_y q_\theta(y)
\log\frac{q_\theta(y)}{p_s(y)}.
$$

求导得到

$$
\begin{aligned}
\nabla_\theta J_s
&=\sum_y\nabla_\theta q_\theta(y)
\left[\log\frac{q_\theta(y)}{p_s(y)}+1\right] \\
&=\sum_yq_\theta(y)u_\theta(y)
[\Delta_\theta(y)+1].
\end{aligned}
$$

利用 score 的零均值恒等式 $\sum_yq_\theta(y)u_\theta(y)=0$，有

$$
\boxed{
\nabla_\theta J_s
=\mathbb E_{y\sim q_\theta}
[\Delta_\theta(y)u_\theta(y)]
}.
$$

仓库中的 teacher reward 是

$$
r_\theta(y)=\log p_s(y)-\log q_\theta(y)=-\Delta_\theta(y),
$$

而 opd*noratio 的 policy loss 是 $-r*\theta(y)\log q\_\theta(y)$，其中 reward/advantage 在进入 policy loss 前被 detach。因此它产生的梯度贡献为

$$
-r_\theta(y)u_\theta(y)
=\Delta_\theta(y)u_\theta(y),
$$

方向与最小化 reverse-KL 一致。对应代码见 [compute_token_reward_direct_advantage](../verl/verl/trainer/ppo/opd_core_algos.py) 与 [compute_policy_loss_opd_noratio](../verl/verl/trainer/ppo/opd_core_algos.py)。

### 1.3 完整序列梯度不是默认 estimator

如果目标是整条序列上的 OPD 缺口，因果状态使一个 token 的 score 还会影响后续 token 的状态与 log-prob。写成序列级 score-function 形式，完整梯度包含后缀缺口：

$$
\nabla_\theta J_{\mathrm{seq}}
=
\mathbb E\left[
\sum_{t=1}^{T}
\left(
\sum_{t'=t}^{T}\Delta_{t'}
\right)u_t
\right].
$$

当前默认 estimator 使用的是

$$
\hat g_{\mathrm{direct}}
=\frac{1}{M}
\sum_{r,\ell}
\Delta_{r,\ell}u_{r,\ell},
$$

其中 $M$ 是 token-mean 下的有效 token 数。它提供 dense、immediate 的信号，方差和实现成本较低；代价是不能直接当作上面的完整序列梯度。仓库还实现了 discounted_token_reward，它沿 response 计算 reward-to-go，但这仍应与完整序列 OPD 推导区分。相关实现见 [verl/verl/trainer/ppo/opd_core_algos.py](../verl/verl/trainer/ppo/opd_core_algos.py)。

如果不先区分这两个 estimator，就可能把 estimator bias 错认为 Adam、Batch Size 或 Fisher 谱的效果。Cai et al. 2026[^cai26] 讨论完整 OPD 梯度时，使用的正是这种后缀累计结构；本文对当前默认训练路径只采用 immediate-token 版本。

### 1.4 LOG_PROB_TOP_K 与 METRIC_TOP_K

当前代码把两个 top-$k$ 开关设计成不同的实验变量：

| 开关           | K>0 时发生什么                                                                                        | 是否改变训练目标 |
| -------------- | ----------------------------------------------------------------------------------------------------- | ---------------- |
| LOG_PROB_TOP_K | teacher reward 变为 (B,T,K) 的 top-$k$ 候选 reward，默认切到 vanilla_topk                             | 是               |
| METRIC_TOP_K   | 计算 student/teacher top-$k$ overlap、mass、reward 等指标，reward 仍是 (B,T) sampled-token reverse KL | 否               |

若两者同时打开，LOG_PROB_TOP_K 优先；如果只想观察 top-$k$ 对训练信号的影响，应保持 LOG_PROB_TOP_K=0 并设置 METRIC_TOP_K>0。当前冻结态脚本正是 0/16，并且 [ray_trainer.py](../verl/verl/trainer/ppo/ray_trainer.py) 明确拒绝 frozen + log_prob_top_k>0。

因此：

- METRIC_TOP_K 的改变应主要出现在观测距离、overlap 与 mass 指标中；
- LOG*PROB_TOP_K 会改变 advantage 的张量形状、candidate support、梯度均值与协方差，理论对象应变成新的 $g*{\mathrm{OPD}}$、$\Sigma*{\mathrm{OPD}}$ 和 $H*{\mathrm{OPD}}$；
- top-$k$ policy-loss 路径不能被称为“只是换了测量口径”。

## 2. OPD Fisher 几何

### 2.1 匹配点的 Hessian

在冻结状态集 $\mathcal S$ 上，假设存在 $\theta^*$ 使得

$$
q_{\theta^*}(\cdot\mid s)=p_s(\cdot)
\quad\text{for the measured states }s.
$$

令学生 logits 对参数的 Jacobian 为 $J_s$，softmax 的输出协方差为

$$
C(p_s)
=\operatorname{diag}(p_s)-p_sp_s^\top.
$$

在 $\theta^*$ 附近，reverse-KL 的局部展开为

$$
\operatorname{KL}
\left(
q_{\theta^*+\delta}(\cdot\mid s)
\Vert q_{\theta^*}(\cdot\mid s)
\right)
=\frac12\delta^\top
J_s^\top C(p_s)J_s\delta
+O(\Vert\delta\Vert^3).
$$

对冻结状态平均，得到

$$
\boxed{
H_{\mathrm{OPD}}
=F_{\mathrm{OPD}}
=\mathbb E_{s\sim\mu_{\mathcal S}}
\left[J_s^\top C(p_s)J_s\right]
}.
$$

这是 reverse-KL 在分布匹配点的局部几何，不是把任意网络 Hessian 经验地替换成 Fisher。Fisher、GGN 与 Hessian 的关系依赖输出分布、参数化和匹配条件；自然梯度的标准讨论见 Martens 2020[^martens14]。

### 2.2 教师失配与 empirical Fisher

如果教师分布不在学生模型族内，令 $\theta^*$ 为冻结目标的局部最优点，实际 Hessian 应写成

$$
H_{\mathrm{OPD}}
=F_{\mathrm{OPD}}+R_{\mathrm{misfit}},
$$

其中 $R_{\mathrm{misfit}}$ 来自非零的 $\log q_{\theta^*}-\log p_s$ 与模型二阶导。缺口较大时，不能把所有尾部都解释成 Fisher 的低特征值。

还要区分三种矩阵：

- **真 Fisher**：按模型分布 $q_\theta$ 对 score 外积取期望；
- **empirical Fisher**：在观测 token 上对样本梯度做外积；
- **Hessian/GGN**：目标对参数的二阶导或输出空间 Gauss--Newton 近似。

在非零失配和有限 token 集上，empirical Fisher 不自动等于真 Fisher，也不自动等于 Hessian。Kunstner et al. 2019[^kunstner19] 专门讨论了这种差异。若要验证本文的局部谱，应优先在固定 response pool 上做 Fisher/GGN HVP，而不是把一个 batch gradient outer product 直接命名为 Fisher。

## 3. 固定状态下的谱吸收

### 3.1 一阶模态递推

令 $\delta_t=\theta_t-\theta^*$。在局部二次模型中

$$
J(\theta^*+\delta)
\approx
J(\theta^*)+\frac12\delta^\top H\delta,
\qquad
\nabla J\approx H\delta.
$$

普通 GD 的更新为

$$
\delta_{t+1}=(I-\eta H)\delta_t.
$$

若 $He_i=\lambda_i e_i$，并写 $\delta_t=\sum_i x_i(t)e_i$，则

$$
x_i(t+1)=(1-\eta\lambda_i)x_i(t).
$$

记 $q_i=1-\eta\lambda_i$。稳定性需要 $\lvert q_i\rvert<1$；当 $0<\eta\lambda_i<1$ 时，各模态单调衰减，才可以把距离近似写成非负模态的混合。Adam 的局部形式更接近

$$
\delta_{t+1}
\approx
\delta_t-\eta P_tH_t\delta_t+\eta\xi_t,
$$

所以固定谱推导还需要 $P_t$、$H_t$ 在观测窗口内变化不大。若 clip_grad=1.0 仍然频繁生效，更新不再是这个线性递推。

### 3.2 L1 advantage 距离

在固定 response pool 上，若 advantage 的模式符号没有大量抵消，可用

$$
d_{\mathrm{L1}}(t)
\approx
\sum_i a_iq_i^t,
\qquad a_i\ge0
$$

表示 $\mathbb E\lvert\Delta\rvert$。于是离散相对吸收率

$$
v_{\mathrm{L1}}(t)
=1-\frac{d_{\mathrm{L1}}(t+1)}{d_{\mathrm{L1}}(t)}
=\eta\sum_iw_i(t)\lambda_i,
$$

其中

$$
w_i(t)
=\frac{a_iq_i^t}{\sum_j a_jq_j^t}.
$$

因此

$$
\boxed{
v_{\mathrm{L1}}(t)
=\eta\langle\lambda\rangle_t
}.
$$

这解释了为什么高曲率方向先消失、剩余吸收率随后下降。对 L1 口径，$a_i\ge0$ 是“符号抵消较弱”的近似，不是一般恒等式；对局部二次 KL，非负模态分解才是严格的。

### 3.3 KL 距离约快一倍

同一参数误差在局部二次模型下给出

$$
d_{\mathrm{KL}}(t)
\approx
\frac12\delta_t^\top H\delta_t
=\frac12\sum_i\lambda_i x_i(0)^2q_i^{2t}.
$$

因此

$$
v_{\mathrm{KL}}(t)
=1-\frac{d_{\mathrm{KL}}(t+1)}{d_{\mathrm{KL}}(t)}
\approx
2\eta\langle\lambda\rangle_{\mathrm{KL},t}
$$

在 $\eta\lambda_i\ll1$ 时成立。KL 的参数误差平方带来约两倍的局部时间尺度，并且额外按 $\lambda_i$ 加权；它不意味着实际日志中的任意 KL 代理都必须精确两倍。

当前仓库常记录 sampled-token 的 CE-H 代理。学生熵的变化可能使 CE-H 出现凸起或相消，因此：

- d_L1、sampled-token reverse-KL、CE、entropy 必须拆开记录；
- 不能仅凭一个 KL 代理曲线反推出 Fisher 谱宽；
- 若凸起在 CE/entropy 拆分后消失，而 L1 没有同步凸起，优先检查距离口径。

### 3.4 自然梯度是白化基线

若 $H=F_{\mathrm{OPD}}$ 且理想预条件器为 $P=F_{\mathrm{OPD}}^{-1}$，则

$$
\delta_{t+1}=(1-\eta)\delta_t.
$$

无噪声时

$$
v_{\mathrm{L1}}=\eta,
\qquad
v_{\mathrm{KL}}
=1-(1-\eta)^2
=2\eta-\eta^2.
$$

这是一条白化参考线，不是说当前 Adam 已经实现自然梯度。任何有效预条件器的 OPD 作用，都应体现在压缩 $P H_{\mathrm{OPD}}$ 的有效谱宽；它不能自动消除低信噪比方向的噪声底。Muon、SOAP 或结构化 Fisher 近似的比较，应使用相同 frozen pool 和相同距离口径。

### 3.5 on-policy 的状态漂移项

设 $d(\theta,\mu)$ 是某个距离在参数 $\theta$ 与状态分布 $\mu$ 下的期望。一步变化可以展开为

$$
d_{t+1}-d_t
\approx
\nabla_\theta d_t^\top(\theta_{t+1}-\theta_t)
+
\left\langle
\frac{\delta d_t}{\delta\mu},
\mu_{t+1}-\mu_t
\right\rangle.
$$

对应的观测吸收率近似为

$$
v_{\mathrm{obs}}(t)
\approx
v_{\mathrm{fixed}}(t)
-
\frac{1}{d_t}
\left\langle
\frac{\delta d_t}{\delta\mu},
\mu_{t+1}-\mu_t
\right\rangle.
$$

frozen 实验把状态与 response token pool 固定，压低第二项；on-policy 实验每步重新生成学生 rollout，第二项重新进入。故

$$
r_{\mathrm{drift}}(t)
=v_{\mathrm{on}}(t)-v_{\mathrm{fixed}}(t)
$$

可以作为状态分布漂移的诊断候选，但只有在 optimizer、距离口径和 checkpoint 对齐后才有解释力。

## 4. OPD 的有效 Batch Size

### 4.1 序列内自相关

令 $h_{r,\ell}$ 为第 $r$ 条 rollout 的第 $\ell$ 个有效 token 梯度贡献，先忽略 response 长度差异，定义

$$
\hat g
=\frac{1}{RL}
\sum_{r=1}^{R}\sum_{\ell=1}^{L}h_{r,\ell}.
$$

如果不同 rollout 近似独立，单条 rollout 内相距 $k$ 的协方差为 $\Gamma_k$，则

$$
\operatorname{Cov}(\hat g)
=\frac{1}{RL^2}
\left[
L\Gamma_0
+2\sum_{k=1}^{L-1}(L-k)\Gamma_k
\right].
$$

若 $\Gamma_k=\rho_k\Gamma_0$，定义有效长度

$$
\boxed{
L_{\mathrm{eff}}
=
\frac{L}{
1+2\sum_{k=1}^{L-1}(1-k/L)\rho_k
}
}.
$$

正相关使 $L_{\mathrm{eff}}<L$。增加 response 长度不一定线性增加有效 Batch，因为相邻状态相似、共享 prefix、后缀 reward 或 response 内的模式重复都可能增大 $\rho_k$。

### 4.2 prompt cluster 设计效应

若每个 prompt 生成 $n$ 条 response，设同一 prompt 内的梯度相关系数近似为 $\rho_{\mathrm{prompt}}$。若 $R$ 记总 rollout 数，prompt cluster 的设计效应为

$$
D_{\mathrm{cluster}}
=1+(n-1)\rho_{\mathrm{prompt}}.
$$

于是

$$
\boxed{
B_{\mathrm{OPD}}
\approx
\frac{R L_{\mathrm{eff}}}{D_{\mathrm{cluster}}}
}.
$$

当前 N*RESPONSES=1，所以 $D*{\mathrm{cluster}}\approx1$，基线近似为 $B_{\mathrm{OPD}}\approx64L_{\mathrm{eff}}$。未来增加 response 数时，总 rollout 数增加并不代表等比例增加独立信息；相关 response 会被 $D_{\mathrm{cluster}}$ 抵消。

response 长度不相等时，上式是便于比较的近似。实际 token-mean 使用有效 token mask 做归一化，应按每条 rollout 的有效 token 权重估计协方差，而不能把 padding 长度放进 $L$。

### 4.3 OPD 版二阶最优步长

令 $\Sigma_{\mathrm{OPD}}$ 表示 token 梯度贡献的协方差尺度，则

$$
\operatorname{Cov}(\hat g)
\approx
\frac{\Sigma_{\mathrm{OPD}}}{B_{\mathrm{OPD}}}.
$$

将科学空间的局部二阶步长结论[^k11260][^k11280][^k11301] 应用于冻结态 OPD，可得

$$
\boxed{
\eta_{\mathrm{OPD}}^*
\approx
\frac{\eta_{\max}}
{1+\mathcal B_{\mathrm{noise}}^{\mathrm{OPD}}/B_{\mathrm{OPD}}}
},
$$

其中

$$
\eta_{\max}
=\frac{g^\top g}{g^\top H_{\mathrm{OPD}}g},
\qquad
\mathcal B_{\mathrm{noise}}^{\mathrm{OPD}}
=\frac{
\operatorname{tr}\left(\Sigma_{\mathrm{OPD}}H_{\mathrm{OPD}}\right)
}{
g^\top H_{\mathrm{OPD}}g
}.
$$

这是固定状态、局部二次和 SGD-like 方向下的参考式。对 Adam，真正的方向是 $m_t/\sqrt{v_t}$，所以不能把这个公式直接当作 Adam 的精确最优学习率；它的用途是说明为什么应按 $B_{\mathrm{OPD}}$ 而不是按 rollout 数对齐实验。

### 4.4 EMA 的有效 Batch

一阶 EMA

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

在跨步噪声独立、统计量变化慢的稳态近似下，将梯度噪声协方差乘以

$$
\kappa(\beta_1)
=\frac{1-\beta_1}{1+\beta_1}.
$$

因此一阶动量的噪声等效 Batch 约为

$$
\boxed{
B_{\mathrm{OPD,EMA}}
\approx
B_{\mathrm{OPD}}
\frac{1+\beta_1}{1-\beta_1}
}.
$$

当前 beta1=0.9 时

$$
\kappa=\frac{0.1}{1.9}\approx0.0526,
\qquad
B_{\mathrm{OPD,EMA}}\approx19B_{\mathrm{OPD}}.
$$

这只是跨步平均带来的方差压缩，不表示 64 条 rollout 真等价于 $64\times19$ 条独立 rollout。对于完整 Adam，归一化分母、$\beta_2$、epsilon、weight decay 和 clipping 都会改变这个解释；更准确的说法是“一阶动量部分具有约 19 倍的独立噪声等效 Batch”。

当前 beta2=0.999 的指数衰减时间常数约为

$$
\tau_{\beta_2}
\approx
\frac{1}{1-\beta_2}
\approx1000
$$

个 optimizer step，半衰期约为 $693$ step。冻结态默认只跑 300 step，因此不能把整个 run 中的 Adam 当作已经收敛的固定预条件器。Adam 的实际行为还需结合 m、v 的逐层统计观察；其基本更新定义见 Kingma & Ba 2015[^kingma15]。

### 4.5 OPD 版 Surge 判据

科学空间的平均场推导[^k11280][^k11301] 给出一个可用于 OPD 消融的启发式判据。定义

$$
\mathcal B_{\mathrm{simple}}^{\mathrm{OPD}}
=\frac{\operatorname{tr}(\Sigma_{\mathrm{OPD}})}{g^\top g},
\qquad
\rho_B
=\left(
1+\frac{\mathcal B_{\mathrm{simple}}^{\mathrm{OPD}}}{B_{\mathrm{OPD}}}
\right)^{-1/2}.
$$

在 Sign/Adam-like 平均场与对角 Hessian 近似下，学习率分母包含

$$
D\left(
\frac{\kappa}{\rho_B}
+(1-\kappa)\rho_B
\right).
$$

其内部极小点为

$$
\rho^*
=\sqrt{\frac{1-\beta_1}{2\beta_1}}.
$$

当 $\beta_1>1/3$ 时，该点位于 $(0,1)$，理论上可能出现先增后减的 Surge 区域；对应的临界有效 Batch 近似为

$$
\boxed{
B_{\mathrm{OPD,crit}}
\approx
\frac{1-\beta_1}{3\beta_1-1}
\mathcal B_{\mathrm{simple}}^{\mathrm{OPD}}
}.
$$

对 beta1=0.9，前面的系数约为 $0.0588$。这是平均场预测，不是 AdamW 的严格定理。实验上应该比较 $B_{\mathrm{OPD}}$ 与这个阈值，而不是直接比较 64 与阈值；同时检查 Surge 是否随 beta2 记忆、clip 频率或距离口径移动。

## 5. OPD 的噪声底

### 5.1 SGD-like 噪声底

沿一个 Fisher 特征方向，写成

$$
x_{t+1}
=(1-\eta\lambda_i)x_t+\eta\xi_t,
\qquad
\mathbb E[\xi_t^2]
=\frac{\sigma_i^2}{B_{\mathrm{OPD}}}.
$$

小步长下的稳态方差为

$$
\operatorname{Var}(x_i)_\infty
\approx
\frac{\eta\sigma_i^2}
{2\lambda_iB_{\mathrm{OPD}}}.
$$

局部 KL 的该模式贡献约为

$$
d_{\infty,i}^{\mathrm{KL}}
\approx
\frac{\eta\sigma_i^2}
{4B_{\mathrm{OPD}}}.
$$

求和后得到

$$
\boxed{
d_\infty^{\mathrm{KL}}
\approx
d_{\mathrm{repr}}
+
\frac{\eta}{4B_{\mathrm{OPD}}}
\sum_i\sigma_i^2
}.
$$

其中 $d_{\mathrm{repr}}$ 汇总教师不可表示、固定状态覆盖不足、top-$k$ 截断和其它不随随机梯度平均消失的缺口。这个结论与 SGD 作为噪声驱动动力学的局部观点一致[^mandt17]。

可检验判据是：

- 固定 optimizer、状态 pool 和距离口径后，增大 $B_{\mathrm{OPD}}$，底近似按 $1/B_{\mathrm{OPD}}$ 下降，支持 SGD-like 噪声底；
- 底基本不变，优先检查 $d_{\mathrm{repr}}$、top-$k$ support、teacher/student 失配；
- 若只在 Adam 下出现近似不随 Batch 变化的软底，继续检查归一化更新的低信噪比机制。

### 5.2 Adam 低信噪比软底

令 $\kappa=(1-\beta_1)/(1+\beta_1)$。在低信噪比坐标上，用平均场把归一化更新近似成噪声驱动的有界坐标递推，可得到启发式关系

$$
\operatorname{Var}(x_i)_\infty
\approx
\frac{\eta\kappa}{2\lambda_i},
\qquad
d_{\infty,i}^{\mathrm{Adam}}
\approx
\frac{\eta\kappa}{4}.
$$

于是低 SNR 模式的总贡献近似为

$$
\boxed{
d_\infty^{\mathrm{Adam}}
\approx
d_{\mathrm{repr}}
+
\frac{\eta\kappa}{4}
N_{\mathrm{low\text{-}SNR}}
}.
$$

这里的曲率抵消是局部平均场结果，不是 AdamW 在 OPD 上的严格稳态定理。$u_i$ 与 $x_i$ 可能相关，$\beta_2$ 尚未稳态，epsilon、weight decay 和 gradient clipping 也未包含。它的实验价值在于提供区分预测：

- SGD-like 底应随 $1/B_{\mathrm{OPD}}$ 明显下降；
- Adam 低 SNR 软底可能主要随 $\eta$、$\beta_1$ 和低 SNR 坐标数量变化；
- 降低 beta2 应主要改变早期瞬态与状态追踪速度，而不必直接改变固定状态的表示缺口。

## 6. frozen 与 on-policy

### 6.1 frozen 实验隔离了什么

[scripts/run_frozen_state_opd.sh](../scripts/run_frozen_state_opd.sh) 设置 algorithm.frozen_rollout=True，并关闭 rollout importance sampling 与 rollout reward shaping。trainer 首次生成的 response pool 被缓存，后续 step 在同一批状态与 response token 上重新计算 student log-prob，再执行 actor 更新。论文中的 frozen-state 对照也用来区分状态供给与参数优化的作用[^fu26]。

这意味着 frozen 实验固定了：

- prompt 与 response token 的比较集；
- response mask 与 response 长度；
- 学生采样状态的经验分布；
- LOG_PROB_TOP_K=0 时的 sampled-token reward 支持。

它没有固定：

- student 当前的 log-prob；
- Adam 的 m_t、v_t；
- 参数位置和局部曲率；
- teacher forward 之外的 student-dependent metric。

所以 frozen 是“固定状态目标上的重复优化”，不是把 optimizer 也冻结。它适合测量固定目标的谱吸收与噪声底，但不能代表正常 on-policy 的状态分布反馈。

### 6.2 on-policy 引入的反馈

正常 on-policy 每个训练 step 都由当前学生生成新的 response。参数更新会改变：

1. 访问到的状态分布 $\mu_t$；
2. token 梯度贡献的相关结构；
3. teacher/student log-prob 缺口的振幅；
4. 重新加权的 Fisher/Hessian。

因此 on-policy 曲线上的变化可以同时来自参数吸收和状态重采样。若没有 frozen 对照，把所有速度变化都归因于优化器是不成立的。

### 6.3 速率差的诊断公式

在相同初始 checkpoint、相同 optimizer 与相同距离定义下，定义

$$
r_{\mathrm{drift}}(t)
=v_{\mathrm{on}}(t)-v_{\mathrm{fixed}}(t).
$$

若 r_drift 与 rollout 的状态/长度分布同步，而 frozen 曲线保持平滑，主要嫌疑是状态漂移；若两者同步出现凸起，且凸起按 optimizer step 而不是 $\eta t$ 移动，应同时检查 beta2 瞬态、gradient clipping 与 CE-entropy 相消。

## 7. 可证伪的 OPD 预测

### P1：同一 frozen pool 上，KL 速率约为 L1 速率的两倍

在同一 response pool 上同时计算

$$
d_{\mathrm{L1}}=\mathbb E\lvert\Delta\rvert,
\qquad
d_{\mathrm{KL}}=\text{sampled-token reverse-KL proxy}.
$$

若局部谱模型成立，预期

$$
v_{\mathrm{L1}}\approx\eta\langle\lambda\rangle,
\qquad
v_{\mathrm{KL}}\approx2\eta\langle\lambda\rangle_{\mathrm{KL}}.
$$

两者不会在任意训练阶段精确保持 2 倍；该关系只在同一局部窗口、同一状态 pool 和小步长下比较。

### P2：按 B_OPD 对齐比按 rollout 数对齐更稳定

改变 $R$、response 长度或每个 prompt 的 response 数，同时记录 $\Gamma_k$、$\rho_{\mathrm{prompt}}$ 与 $B_{\mathrm{OPD}}$。预期：

- 曲线不会只按 rollout 数完全对齐；
- $\eta t$ 负责主要的时间轴折叠；
- $B_{\mathrm{OPD}}$ 更能解释早期方差和末期噪声底；
- R=64 不能作为所有 token 口径下的 Batch Size。

### P3：beta2 改变瞬态，beta1 改变有效噪声与软底

在同一 frozen pool 上比较 (0.9,0.999)、(0.9,0.95)、(0.5,0.5) 与 (0,0)：

- beta2=0.999 的二阶状态追踪应明显慢于 beta2=0.95；
- 如果曲线变化按 optimizer step 的 EMA 记忆尺度移动，而不是按 $\eta t$ 移动，优先归因于 EMA 瞬态；
- 增大 beta1 在独立跨步噪声近似下应降低一阶 EMA 的有效方差，但不保证降低所有 Adam 轨迹的总误差；
- (0,0) 只适合作为 SGD-like/无 EMA 对照，仍需明确其 optimizer 实现与 weight decay 设定。

### P4：METRIC_TOP_K 与 LOG_PROB_TOP_K 的影响必须不同

只改变 METRIC*TOP_K 时，训练梯度、parameter trajectory 与 optimizer state 应基本不变，变化应只出现在 top-$k$ 观测；改变 LOG_PROB_TOP_K 时，reward 和 policy loss 的张量形状、support 与梯度统计都会变化，因而可能改变 $g*{\mathrm{OPD}}$、$\Sigma*{\mathrm{OPD}}$、$H*{\mathrm{OPD}}$ 和谱吸收曲线。

### P5：frozen 与 on-policy 的差值估计状态漂移

在相同 checkpoint 序列上计算 $d_{\mathrm{fixed}}$、$d_{\mathrm{on}}$、$v_{\mathrm{fixed}}$ 与 $v_{\mathrm{on}}$。如果 r_drift 与 rollout 分布变化同步，而 frozen 曲线平滑，支持状态漂移解释；如果二者同步凸起，应优先检查 optimizer transient、距离口径和 clipping。

## 8. 最小测量方案

### 8.1 每步 token 级日志

在 advantage 计算后保存：

- Delta、abs(Delta)、token position、rollout id、prompt id；
- 有效 token mask、response 长度与每条 rollout 的 token 数；
- score 范数或固定随机投影，用于估计 $\Sigma_{\mathrm{OPD}}$；
- LOG_PROB_TOP_K>0 时的 candidate support id、teacher/student mass 与 overlap；
- sampled-token 的 L1、CE、entropy、KL proxy，避免把 CE-H 当作唯一距离。

这些量可以估计序列自相关、prompt cluster 设计效应、$B_{\mathrm{OPD}}$ 与多个距离口径。

### 8.2 冻结态检查点

在 step 30、100、200 附近，对同一 response pool 测量：

1. Fisher/GGN 的随机 HVP 或小规模 Lanczos；
2. Adam m、v 的逐层 RMS 与分位数；
3. d_L1、d_KL、CE、entropy；
4. v_direct、v_geo 与 $\alpha=t v$；
5. top-$k$ support overlap 与 support 改变比例；
6. gradient clipping 触发比例与全局 grad norm。

Fisher/HVP 的工程背景见 [docs/2026-09-06-lanczos-fisher-hvp-survey.md](2026-09-06-lanczos-fisher-hvp-survey.md)；谱吸收的距离口径与边界见 [docs/2026-09-06-absorption-rate-spectral-theory-restructured.md](2026-09-06-absorption-rate-spectral-theory-restructured.md)。

### 8.3 最小消融矩阵

| 消融 | 固定项                        | 改变项                | 主要判据                                     |
| ---- | ----------------------------- | --------------------- | -------------------------------------------- |
| A    | response pool、optimizer、lr  | $R$ 或 token 子集大小 | $B_{\mathrm{OPD}}$ 是否比 $R$ 更能解释噪声底 |
| B    | response pool、$R$、lr        | beta2=0.999 对 0.95   | 瞬态是否按 EMA 记忆长度移动                  |
| C    | response pool、$R$、betas     | lr 乘 0.5、1、2       | $\eta t$ 折叠是否成立                        |
| D    | response pool、$R$、optimizer | METRIC_TOP_K          | 只改变观测，不改变训练轨迹                   |
| E    | response pool、$R$、观测口径  | LOG_PROB_TOP_K        | 训练目标改变是否改变有效谱                   |
| F    | 相同 checkpoint 与 optimizer  | frozen 对 on-policy   | $r_{\mathrm{drift}}$ 是否解释速率差          |

## 9. 结论与边界

### 9.1 可以直接复用的结论

1. 冻结态 OPD 的局部 Hessian 在可表示匹配点是 $F_{\mathrm{OPD}}$，由 student token 分布和模型 Jacobian 决定。
2. token_reward_direct 是 immediate-token estimator；完整序列后缀累计梯度不能混入当前默认路径的解释。
3. L1 advantage 与 KL 距离对同一参数误差具有不同谱权重；局部 KL 速率约为 L1 的两倍，但实际 CE-entropy 代理会污染这个关系。
4. OPD 的有效 Batch Size 由 token 数、序列内相关性和 prompt cluster 共同决定；data.train_batch_size=64 只是当前 rollout 计数。
5. beta1=0.9 在独立噪声近似下给一阶 EMA 约 19 倍有效 Batch 放大，但 beta2=0.999 在 300 step 的短 run 中仍是长瞬态。
6. LOG_PROB_TOP_K 是训练目标变更，METRIC_TOP_K 是观测变更。
7. frozen 对照是分离参数优化与 on-policy 状态漂移的最低成本实验。

### 9.2 不能直接声称的内容

- 不能只凭一条 $d_t$ 曲线反推出完整 Hessian/Fisher 谱；
- 不能把 immediate-token estimator 当作完整序列 OPD 梯度；
- 不能把 64 条 rollout 当成 64 个独立 token 样本；
- 不能把 AdamW 的 v_t 当成已经收敛的固定 Fisher 预条件器；
- 不能把 sampled-token CE-H 的局部凸起直接解释成 Fisher 谱宽；
- 不能在没有 frozen 对照时，把 on-policy 曲线变化全部归因于参数优化；
- 不能把 top-$k$ policy-loss 路径与 metric-only top-$k$ 观测当成同一实验。

最短的研究闭环是

$$
\text{token gradient log}
\longrightarrow
(B_{\mathrm{OPD}},\Sigma_{\mathrm{OPD}})
\longrightarrow
\text{frozen Fisher/HVP}
\longrightarrow
\text{预测 }d(t),v(t),d_\infty
\longrightarrow
\text{on-policy 残差}.
$$

### 参考文献

[^k11260]: 科学空间. 《重新思考学习率与 Batch Size（一）：现状》. 2025. [原文](https://kexue.fm/archives/11260)

[^k11280]: 科学空间. 《重新思考学习率与 Batch Size（二）：平均场》. 2025. [原文](https://kexue.fm/archives/11280)

[^k11301]: 科学空间. 《重新思考学习率与 Batch Size（四）：EMA》. 2025. [原文](https://kexue.fm/archives/11301)

[^cai26]: Y. Cai, D. Cao, L. Lin, et al. _Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation_. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)

[^fu26]: Z. Fu, B. He, Y. Zuo, et al. _Rethinking On-Policy Distillation of Large Language Models II: One Training Example_. arXiv 2026. [arXiv:2609.04172](https://arxiv.org/abs/2609.04172)

[^kingma15]: D. Kingma, J. Ba. _Adam: A Method for Stochastic Optimization_. ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

[^martens14]: J. Martens. _New Insights and Perspectives on the Natural Gradient Method_. JMLR 21(232), 2020. [arXiv:1412.1193](https://arxiv.org/abs/1412.1193)

[^kunstner19]: F. Kunstner, P. Hennig, L. Balles. _Limitations of the Empirical Fisher Approximation for Natural Gradient Descent_. NeurIPS 2019. [arXiv:1905.12558](https://arxiv.org/abs/1905.12558)

[^mandt17]: S. Mandt, M. Hoffman, D. Blei. _Stochastic Gradient Descent as Approximate Bayesian Inference_. JMLR 18(134), 2017. [arXiv:1704.04289](https://arxiv.org/abs/1704.04289)
