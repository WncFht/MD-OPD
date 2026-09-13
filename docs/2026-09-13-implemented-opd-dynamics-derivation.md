# OPD 三个变体的实现动力学推导：SGD / Adam / MDPO

> 目标：把代码里**实际在跑**的三个变体（vanilla SGD OPD、Adam OPD、MD-OPD）在给定 config 下的权重动力学写成闭式，给出可检验的定量预测。推导框架沿用 EffOPD F.5[^effopd] 的局部二次模型与 GeoSD[^geosd] 的近端项展开，但落到实现层面：有限步内层滤波、ratio surrogate 的不定曲率、k3+ 惩罚的真实梯度语义、clip/AdamW/weight-decay 的修正项。
>
> 实现事实全部核对自 `verl/verl/workers/utils/losses.py:164-192`、`verl/verl/trainer/ppo/core_algos.py:1286,2187-2212,2215-2250`、`verl/verl/trainer/ppo/opd_trainer_mixin.py:85-101` 与各 run 的 `plots/runs/*__meta.json`。

## 0. Config 事实（推导的前提条件）

所有 one-shot 臂共享（`t22.parquet` 单样本，300 个 outer step）：

| 项                    | 值                                                                                    | 推论                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 数据                  | 1 个 prompt；每 outer step 采 $B{=}64$ 条 rollout（n=1，T=1.0）                       | 每步的 frozen batch = 同一题的 64 条独立轨迹                                                            |
| `ppo_mini_batch_size` | 64 = `train_batch_size`                                                               | 每个 epoch 恰好 1 个 minibatch → **每 outer step 的内层梯度步数 = `ppo_epochs` $m$**，无 minibatch 切分 |
| 优化器                | AdamW，$\alpha{=}10^{-6}$ 恒定无 warmup，$\beta{=}(0.9,0.999)$，$\lambda_{wd}{=}0.01$ | 每步 decay 因子 $1-10^{-8}$，可忽略；见 §3                                                              |
| `grad_clip`           | 1.0（全局范数）                                                                       | 方向不变、幅值封顶；clip 与否见指标                                                                     |
| PPO clip              | $\varepsilon{=}0.2$（low/high 对称）                                                  | log 域界 $\ell\in[\log0.8,\log1.2]=[-0.223,+0.182]$                                                     |
| 聚合                  | `token-mean`                                                                          | 所有系数含 $1/M$（M = batch 内 token 数）                                                               |
| MDPO                  | `mdpo_kl_loss=True`，$\eta_t$ 线性 $1\to0.01$；`kl_loss_type=k3+`                     | $\eta_t = 1-0.99\,t/299$；系数 $1/\eta_t$：$1\to100$                                                    |

各臂的锚语义（`frozen_rollout` 与 inner loop 不兼容，代码第 204 行）：

| run        | $m$ | 锚 $\mu$                        | 损失                                           | 结局                            |
| ---------- | --- | ------------------------------- | ---------------------------------------------- | ------------------------------- |
| `5aub1k4h` | 1   | 每步新 rollout $\pi_{\theta_k}$ | clipped ratio                                  | $d_{last}=0.00154$              |
| `nh3jj9bt` | 4   | 每步新 rollout                  | clipped ratio                                  | $d_{last}=0.00075$              |
| `zyskh7f2` | 16  | 每步新 rollout                  | clipped ratio                                  | $d_{last}=0.00409$（更差）      |
| `6p579hnn` | 4   | 每步新 rollout + k3+ 弹簧       | clipped ratio + $(1/\eta_t)\cdot\frac12\ell^2$ | $d_{last}=0.00109$，熵 0.632    |
| `ceejzu26` | 1   | **$\pi_0$ 冻结 300 步**         | noratio $-a\log p$                             | $d_{end}/d_{30}=0.012$（最深）  |
| `0i38mr1k` | 1   | **$\pi_0$ 冻结 300 步**         | clipped ratio                                  | 崩盘 $d\approx0.50$，failed@218 |

实测指标（`plots/runs/*__history.csv`）：`pg_clipfrac` = 4ep 0.1%、16ep 4.6%、MDPO 0.03%、frozen-vanilla **23%**；`ppo_kl`（每内层步的平均 $-\ell$）= 4ep ~$10^{-4}$、16ep $5{\times}10^{-3}$、MDPO ≈0、frozen-vanilla 0.04–0.06。**clip 在 lr $10^{-6}$、$m\le16$ 下基本不活跃**——下面的推导中 clip 按" rarely 触墙"处理。

## 1. 局部二次模型

第 $k$ 个 outer step：rollout 策略 $\mu_k=\pi_{\theta_k}$ 生成 batch $\mathcal B_k$（冻结的 states $s$、采样 token $y$、冻结的 advantage $a_i=\log q_i-\log\mu_i-b_i$）。内层位移 $\delta=\theta-\theta_k$。真实目标

$$
J(\theta_k+\delta)\approx J_k+g^\top\delta+\tfrac12\delta^\top H\delta,
\qquad
H=\mathbb E_{s}\bigl[J_s^\top C(\mu_k)J_s\bigr]+\text{residual terms},
$$

$g=\nabla J(\theta_k)=\mathbb E[s_i\,\ell_i^{res}]$，$\ell_i^{res}=\log\mu_i-\log q_i$。$H$ 即 F.5 的 $A$；残余曲率项在 $\mu\approx q$ 附近消失、远离时不可忽略（`docs/MD-NGD/OPD加KL是否等于一步NGD.md` §7）。距离读数 $d$ 是 $\delta$ 的二次型：$d(\delta)\approx\sum_i\lambda_i x_i^2$（$x_i$ 为特征坐标）。

**d-空间速率约定**：记 $\rho_i$ 为一个内层梯度步对 mode $i$ 的 $d$-贡献收缩率。GD 下 $\rho_i=2\alpha\lambda_i+O(\alpha^2\lambda^2)$（参数空间收缩 $(1-\alpha\lambda_i)$，$d$ 是二次量故平方）。NNLS 反演的 `spectrum__ours_sampled_token_kl.csv`（64 modes，$\rho\in[0.004,0.1]$）是 **$d$-空间、$m{=}1$、Adam 预条件之后**的有效谱——反推原始 Hessian 特征值须除有效步长：$\lambda_i^{eff}=\rho_i/2\alpha\in[2{\times}10^3,\,5{\times}10^4]$。

## 2. 变体 A：SGD + noratio（理想参照，ceejzu26 的欧氏版）

损失 $L=-\frac1M\sum_i a_i\log p_i$，梯度 $\hat g=-\frac1M\sum a_i s_i$，$\mathbb E\hat g\approx g$。二次模型上 $m$ 步：

$$
\boxed{\;\delta_m=-\alpha\sum_{r=0}^{m-1}(I-\alpha H)^r\,\hat g
=-\bigl[I-(I-\alpha H)^m\bigr]H^\dagger g+\text{噪声}\;}
$$

mode $i$ 的每 outer step 收缩因子（向不动点 $x_i^*=-g_i/\lambda_i$ 的完成度）：

$$
\boxed{\;\phi_i(m)=1-(1-\alpha\lambda_i)^m\;}
$$

- $m{=}1$：$\phi_i=\alpha\lambda_i$——vanilla，速率正比于 $\lambda_i$，快慢 mode 差 25 倍（实测 $\rho\in[0.004,0.1]$）；
- $m{>}1$ 且 $\alpha\lambda_i\ll1$：$\phi_i\approx m\alpha\lambda_i$——**线性区近似 $m$ 倍加速**；
- $\alpha\lambda_i\to1$：饱和；$\alpha\lambda_i>1$：$(1-\alpha\lambda_i)^m$ 变号→过冲振荡；$>2$：发散。

数值（$\alpha\lambda_i=\rho_i/2\in[0.002,0.05]$，全部远离不稳定区）：

| $\rho_i$ | $\phi_i(1)$ | $\phi_i(4)$ | $\phi_i(16)$ |
| -------- | ----------- | ----------- | ------------ |
| 0.004    | 0.004       | 0.016       | 0.062        |
| 0.03     | 0.03        | 0.113       | 0.38         |
| 0.10     | 0.10        | 0.34        | 0.81         |

**噪声 floor**（每内步注入 $\xi_j$，$\mathrm{Var}_i=\sigma_i^2$）：

$$
\mathrm{Var}[x_i]_m=\frac{\alpha^2\sigma_i^2\,m\;[\,1-(1-\alpha\lambda_i)^{2m}\,]}{1-(1-\alpha\lambda_i)^2}
\;\xrightarrow{\;\alpha\lambda_i\ll1\;}\;
\frac{\alpha\sigma_i^2}{2\lambda_i}
\quad\Longrightarrow\quad
d_\infty=\frac{\alpha}{4B_{\!eff}}\sum_i\sigma_i^2 .
$$

内层步数 $m$ 不改变稳态方差（信号与噪声同比例累积）——**多 epoch 在纯二次模型下只加速、不抬高 floor**；实测 16-epoch 更差，说明真实机制在二次模型之外（§4.2）。

## 3. 变体 B：AdamW（所有 run 的实际优化器）

更新 $u_j=-\alpha\,\hat m_j/(\sqrt{\hat v_j}+\varepsilon)-\alpha\lambda_{wd}\theta_j$。准静态近似 $\hat v_i\approx\mathbb E\hat g_i^2$ 下分两个 regime：

| regime   | 条件                      | 等效更新                                | 谱后果                                                      |
| -------- | ------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| 信号主导 | $\|\bar g_i\|\gg\sigma_i$ | $u_i\approx-\alpha\,\mathrm{sign}(g_i)$ | 每坐标匀速运动；mode 衰减从指数变**线性**，慢 mode 相对加速 |
| 噪声主导 | $\bar g_i\approx0$        | $u_i\approx-\alpha\,\hat g_i/\sigma_i$  | 逐坐标 SNR 白化（噪声大的坐标步长自动缩小）                 |

有效线性算子是 $I-\alpha D^{-1}H$，$D=\mathrm{diag}(\sqrt{\hat v})$——即把谱换成 $D^{-1/2}HD^{-1/2}$ 的特征值。只要 $\mathrm{diag}(H)$ 的分散度小于谱分散度（通常成立），**Adam 已经做了一半的速率均衡**：这解释了为什么 MDPO 相对 4-epoch 臂的增益是边际的——"$m$ 步 Adam"本身已含部分白化。

三个可检验的 Adam 专属预言：

- **A1（后期加速签名）**：快 mode 吸空后 $\hat v_i$ 沿信号方向收缩 → 有效步长 $\alpha/\sqrt{\hat v_i}$ 增大 → 后期实测衰减速率应**超过**静态谱外推。实测：$m{=}1$ 臂 $d_{300}/d_{first}=0.0149$，静态谱前向预测 0.098——实测快 ~6×。与"Adam 自适应白化"一致（也有跨 run 借用谱的误差成分，见 §5 注）。
- **A2**：$\beta_2{=}0.95$ 臂（`tjjxnvw9`）$\hat v$ 窗口 ~20 步 → 预条件器噪声大 → floor 应偏高。可直查尾部深度。
- **A3**：$m{>}1$ 时 $\hat v$ 在内层累积 surrogate 梯度而非真实梯度——预条件器被 surrogate 方向的方差塑形，是 16-epoch 退化的次要机制。

weight decay 与 grad clip：每步 $(1-10^{-8})$ 收缩可忽略；grad clip=1 仅在 $\|\hat g\|_2>1$ 时封顶（早期可能触发，clipfrac 指标可旁证）。

## 4. 变体 C：clipped ratio surrogate（无锚，5aub1k4h/nh3jj9bt/zyskh7f2/0i38mr1k）

损失 $L=-\frac1M\sum_i a_i\,\rho_i\cdot\mathbb 1[\text{unclipped}]$，$\rho_i=e^{\ell_i}$，$\ell_i=\log p_i-\log\mu_i$。

### 4.1 隐性曲率与指数反馈

未触 clip 时梯度 $=-\frac1M\sum a_i e^{\ell_i}s_i$：**有效 advantage $\tilde a_i=a_i e^{\ell_i}$ 随已实现位移指数放大**。surrogate 的隐性 Hessian

$$
H_{surr}=-\mathbb E\bigl[a_i\,e^{\ell_i}\,s_i s_i^\top\bigr]\;\;(\text{不定}),
$$

沿改进方向（$a_i e^{\ell_i}>0$ 加权方向）是**负曲率**——内层迭代自加速，无内部不动点。这是一个结构性事实：**ratio surrogate 的"任务项"自带沿下降方向的正反馈**，它没有局部极小，唯一边界是 clip 墙（$\ell\approx+0.18/-0.22$）与锚弹簧。

### 4.2 三种锚下的命运

- **每步新锚（online，$m{=}1$）**：$\ell$ 每 outer step 归零，反馈来不及启动 → 接近理想 PG。`5aub1k4h`。
- **每步新锚，$m{=}4$–16**：前 1–2 步在 $\ell\approx0$ 线性区；第 3+ 步起 $e^{\ell}$ 权重偏向已动 token（赢家放大），且 surrogate 线性外推与真实 KL 的偏差 $O(\|\delta\|^2)$ 随内层位移累积——**后期内层步的收益递减、且方向偏离真实下降方向**。"陈旧 surrogate"的精确含义 = 在漂移后的点继续优化锚点处的一阶模型。实测 16-epoch clipfrac 4.6%、ppo_kl $5{\times}10^{-3}$（4-epoch 的 25 倍）= 内层确实在走远；但 $d_{last}$ 反而更差——**超出的位移没有转化成 KL 下降**，与"线性方向走过头"一致。
- **冻结锚 $\pi_0$（0i38mr1k）**：$\rho=e^{\ell}$ 跨 outer step 无重置地指数增长；$a_i>0$ token 的力 $\propto e^{\ell_i}$ 单向棘轮，$a_i<0$ 的力随 $\rho\to0$ 自灭——质量单向涌向高-$a$ token → mode 坍缩。实测 clipfrac 23%、ppo_kl 0.04–0.06、$d\to0.50$ 全对上。对照同锚 noratio（`ceejzu26`，力有界 $-a_i s_i$）反而吸得最深——**崩盘不是 OPD 的问题，是 ratio surrogate × 冻结锚这一对组合的问题**。

## 5. 变体 D：MDPO（6p579hnn 的精确实现）

内层目标（token-mean，实现自 `losses.py:170-180`）：

$$
P(\delta)=\underbrace{-\frac1M\sum_i a_i e^{\ell_i}[\text{clip}]}_{\text{任务 surrogate}}
+\underbrace{\frac1{\eta_t}\cdot\frac1M\sum_i\tfrac12\ell_i^2}_{\text{k3+ 的真实梯度项}} .
$$

注意 `kl_loss_type=k3+` 的语义（`core_algos.py:2199-2212`）：forward 值是 k3 估计 $e^{-\ell}+\ell-1$（≈ $\mathrm{KL}(\mu\|p)$ 的估计），但**backward 走的梯度是 $\frac12\ell^2$ 的梯度**（straight-through）。即真正进优化器的惩罚是逐 token 的**平方对数比弹簧**，期望形式 $\frac12\mathbb E_\mu[\ell^2]$，其在锚点的 Hessian $=\mathbb E_\mu[ss^\top]=F$（经验 Fisher）。这是 Kimi OPMD/PMD-mean[^pmdmean] 同族的 squared-log-ratio 罚。

### 5.1 逐 token 力学

第 $i$ 个采样 token 的内层力：$f_i=a_i e^{\ell_i}-\ell_i/\eta_t$（忽略归一化耦合）。不动点方程

$$
\ell_i=\eta_t\,a_i\,e^{\ell_i}
\;\Longleftrightarrow\;
\ell_i e^{-\ell_i}=\eta_t a_i .
$$

左端最大值为 $1/e$（在 $\ell_i{=}1$ 处）：

- $\eta_t a_i>1/e$（早期 $\eta{\approx}1$、或强信号 token）：**无内部不动点**——弹簧压不住指数任务力，token 被推到 clip 墙停下；实现的是 **clip 截断的指数倾斜** $\Delta\ell_i\approx\pm0.2$，而不是几何混合的 $\eta a_i$。
- $\eta_t a_i<1/e$（后期 $\eta{\to}0.01$，或弱信号 token）：内部解 $\ell_i^*\approx\eta_t a_i(1+\eta_t a_i+\cdots)$——近乎冻结的微小倾斜。

所以在"内层收敛"的假想极限下，MDPO 实现的是**封顶版几何混合**：$p^+\propto\mu e^{\min(\eta_t a_i,\,\text{clip})}$；只有当 $\eta_t a_i$ 全部低于阈值时才退化为教科书 EG $p^+\propto\mu^{1-\eta_t}q^{\eta_t}$。

### 5.2 参数空间的内层滤波（实际 regime：远未收敛）

$\alpha{=}10^{-6}$ 下每内步 $|\Delta\ell|\sim\alpha\|\nabla_\theta\ell\|\sim10^{-5}$–$10^{-4}$，4 步远低于不动点（clipfrac 0.03% 证实）。真实动力学是截断滤波：

$$
\delta_m=-\bigl[I-(I-\alpha B)^m\bigr]B^\dagger g_{\!eff},
\qquad
B=H_{surr}+\frac1{\eta_t}F .
$$

（Adam 版把 $I-\alpha B$ 换成 $I-\alpha D^{-1}B$。）三条直接推论：

1. **锚项在 $O(m\alpha)^2$ 阶才起作用**：弹簧梯度 $\propto\ell$，起点 $\ell{=}0$ 时为零——第 1 步就是 vanilla PG，第 2 步起才有回拉。小步极限下
   $\delta_m\approx -m\alpha\,g+\frac{(m\alpha)^2}{2\eta_t}F g/\|g\|$-型修正——**4 步 Adam 实现的 MDPO 离"几何混合"还很远，观测到的差异主要来自内层轨迹的二阶回拉与赢家放大的抑制**。
2. **内层不完全解有谱偏置**：滤波器先完成 $B$ 的大特征值方向——即高 Fisher（高概率/高曲率）token 的弹簧先收敛，低 Fisher 方向滞后 → 残余位移系统性偏向低曲率 mode。可用 E2 的频率十分位分解直接检验。
3. **$\eta_t$ 的语义（收敛极限）**：分布空间里每 outer step 把采样 token 的 log-残差乘 $(1-\eta_t)$——$\eta_t$ 是**分布空间的收缩因子**，$\sum_t\eta_t\approx150$ 意味着收敛解会把 log 距离压 $e^{-150}$ 量级；现实被内层完成度限死。schedule 方向 $1\to0.01$ = 分布空间步长退火——**在有噪声 floor 的问题里这是正确方向**（步长随残余信号衰减），与 homotopy-PMD[^homotopic] 的"衰减正则化→增大步长"是不同目标下的不同最优（他们假设精确内解、凸问题；我们面对采样噪声 floor）。
4. **锚每 outer step 重置** ⇒ prox 点的收缩因子 $\eta/(1{+}\eta)$ 不累积成偏差，外层不动点仍是 $p{=}q$（可达集内）。

### 5.3 熵保持的机制读数

三个叠加：①弹簧对所有已采样 token 反向回拉，净位移小（ppo_kl≈0）；②锚重置使任务 surrogate 的指数反馈每步重启，赢家放大被截断在 $m{=}4$ 步内；③k2 弹簧在 log 域对称、且按采样频率加权——高频 token（教师学生都支持的主干）被钉得最牢，质量迁移主要发生在中频区。**预言**：MDPO 臂的 per-token $\ell$ 分布应比 4-epoch 臂更紧、更对称——可用 token 级 log-ratio 直方图直接检验。

## 6. 数值预测 vs 实测

用反演谱（`mdxt04lm`，全量在线臂）前向模拟各 $m$ 的纯二次动力学 $d_m(t)=\sum_k c_k e^{-m\rho_k t}$：

| 量                   | 模型预测                                   | 实测                     | 判读                                                                            |
| -------------------- | ------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------- |
| $m{=}1$ 臂 $d_{300}$ | 0.0040（谱前向）                           | 0.00154                  | 实测快 ~2.6×：Adam 自适应白化（A1）/跨 run 借谱误差                             |
| $m{=}4$ $d_{300}$    | 0.00006 纯谱；+floor(8e-4) → **0.00086**   | 0.00075                  | **吻合**：4-epoch ≈ "快速吸收 + ~8e-4 floor"                                    |
| $m{=}16$ $d_{300}$   | →0                                         | 0.00409                  | 模型完全失败：内层 5–16 步净贡献 ≈0 甚至为负——surrogate 陈旧/不定曲率的量化证据 |
| MDPO $d_{300}$       | —                                          | 0.00109                  | 与 4ep 同量级、略逊；但见 §5.2-1：差异本就该是二阶小量                          |
| MDPO 后期 $v_t$      | $\approx\eta_t\bar\rho\to 10^{-2}\bar\rho$ | $v_t\to0^-$（step 200+） | 定性吻合：$\eta_t$ 退火到 1% 时收缩力 ≈ floor 噪声回推                          |

**一句话**：纯二次模型在 $m{=}4$ 内定量成立，$m{=}16$ 被数据否决——**内层 epoch 数的收益在 $\sim$4–8 之间被 surrogate 非二次性截断**；MDPO 的增益不是"到达了几何混合"（步长太小到不了），而是"弹簧约束使内层轨迹更高质量"。

## 7. 可检验预测清单

| #   | 预测                                                                      | 检验途径                                           | 现状     |
| --- | ------------------------------------------------------------------------- | -------------------------------------------------- | -------- |
| P1  | $\phi_i(m)=1-(1-\rho_i)^m$：4-epoch 各 mode 完成度可查表                  | 用 E2 分解看残余是否集中在预测的低-$\rho$ mode     | 待 E2    |
| P2  | 内层不完全解的谱偏置：残余偏向低 Fisher（稀有/低曲率）token               | E2 频率十分位×残余                                 | 待 E2    |
| P3  | Adam 后期加速：实测 $v_t$ 尾部高于静态谱外推                              | 已有初步迹象（§6 行 1）；对 history 做局部斜率检验 | 部分支持 |
| P4  | β2=0.95 → floor 更高                                                      | `tjjxnvw9` 尾部深度 vs 同 lr 臂                    | 可查     |
| P5  | 16-epoch 的内层 pg_loss 继续降而真实 $d$ 停滞——"surrogate-真实 gap"       | history.csv 内层指标 vs opd/token_reward_mean      | 可查     |
| P6  | MDPO 后期有效收缩率 ≈ $\eta_t\bar\rho$；若改常数 $\eta$ 后期应不同        | E5 消融                                            | 待跑     |
| P7  | frozen+ratio 崩盘由 $e^\ell$ 棘轮驱动：ppo_kl/clipfrac 应持续上升直至失败 | `0i38mr1k` 已验证（23%/0.05）                      | 成立     |
| P8  | floor $\propto\alpha$（噪声 floor 线性于 lr）                             | E1 退火探针；M4 各臂 floor 深度比                  | 待 E1    |
| P9  | k2 弹簧对称、按 $\mu_i$ 频率加权：高 $\mu$ token 位移最小                 | token 级 $\ell$ 直方图                             | 待 E2    |

## 8. 与 F.5 / GeoSD 推导的关系

- **F.5（Cai et al.[^effopd]）**：在同一二次模型上给 GD vs NGD 的理想轨迹（每步 $F^{-1}$）。本文把它落到实现：有限 $m$ 的谱滤波、ratio surrogate 的不定 $H_{surr}$、k3+ 弹簧的真实梯度语义、clip/AdamW 修正——回答的是"**代码实际跑出来的动力学**离理想轨迹差多远、差在哪"。
- **GeoSD（App A[^geosd]）**：prox 项经 NG 变 checkpoint pullback $-2\eta\lambda F_k^{-1}F_{ckpt}\Delta$；与本文 $B=H_{surr}+F/\eta_t$ 的内层算子是同一族（Fisher 度量的锚），差别在锚滞后（64 步 vs 每步）与进入方式（惩罚项 vs prox 算子）。
- **共同边界**：三者都在 $p\approx q$ 或 GN 近似下成立；远离教师时 $H$ 含残余曲率项，$H\approx F$ 失效——谱模型与所有滤波公式在该 regime 只有定性意义。

### 参考文献

[^effopd]: Y. Cai et al. Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation. arXiv 2026. [arXiv:2605.11739](https://arxiv.org/abs/2605.11739)（附录 F.5：OPD 局部二次模型上的 GD/NGD 动力学）

[^geosd]: I. Jukić, I. Titov. Geometric Self-Distillation for Reasoning Generalization. arXiv 2026. [arXiv:2607.06855](https://arxiv.org/abs/2607.06855)（App A：Fisher 球面、checkpoint pullback 推导）

[^pmdmean]: Y. Xu et al. Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training. arXiv 2026. [arXiv:2602.05933](https://arxiv.org/abs/2602.05933)（squared-log-ratio 罚 = 隐式 KL–χ² 混合正则）

[^homotopic]: X. Li, G. Lan, T. Zhao. Homotopic Policy Mirror Descent. arXiv 2022. [arXiv:2201.09457](https://arxiv.org/abs/2201.09457)
