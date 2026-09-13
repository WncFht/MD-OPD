# OPD 论文与 LLM 参数几何工具研究笔记

更新时间：2026-09-09。检索范围主要为 2024--2026 年的 OpenReview 论文，并用 arXiv、OpenAlex、Crossref 做交叉核对。本文以 `/Users/fanghaotian/Downloads/translate_zh_CN/neurips_2026.tex` 为主线，重点解读附录 F.5（源文件第 803--1080 行附近的“OPD 动力学的局部几何视角”）。

## 1. 一句话结论

这篇论文的实证主张是：相对于 PPO、GRPO、DAPO 等 RL，OPD 在较早阶段就把更新集中到更有功能效用的模块和更稳定的低维方向；因此可以用早期方向外推来加速训练。F.5 给出的理论解释是：在基础模型附近把 logits 对参数线性化，并把 token 级反向 KL 局部二次化，于是得到由

$$
A=\mathbb{E}_c[J_c^\top F_cJ_c]
$$

控制的局部 Gauss--Newton / pullback-Fisher 二次问题。这个解释很有启发性，但不是对真实 LLM 全局训练轨迹的定理：它依赖局部线性化、教师与基础模型足够接近、同策略上下文分布近似固定，以及驱动项 $b$ 的低维集中等假设。

## 2. 论文的逻辑地图

### 2.1 研究问题与比较对象

论文共享同一基础初始化 $W_{\mathrm{Base}}$，比较 RL 和 OPD 的参数位移

$$
\Delta W=W_{\mathrm{trained}}-W_{\mathrm{Base}}.
$$

RL 侧包括 PPO、GRPO、DAPO；OPD 侧使用与学生能力和思维模式相容的 RL 教师。实验覆盖 Qwen2.5/Qwen3，约 1.5B--32B 参数，任务包括数学和代码推理。

### 2.2 性质 1：功能冗余规避

论文先做“同范数比较”：将最终更新写成 $W_{\mathrm{Base}}+\alpha\Delta W$，调节 $\alpha$ 以匹配 RL 与 OPD 的更新范数。OPD 在相同更新预算下的推理收益更高，作者据此认为 RL 的更新中有更多与任务弱相关的分量。

随后使用两类模块干预：

1. **Embedding replacement**：将 RL/OPD 模型的 embedding 换回基础模型，推理影响很小，说明 embedding 更新通常不是主要能力增益来源。
2. **Sliding-window intervention**：对连续层窗口注入某种方法的局部更新，窗口中心左右各 8 层，最大宽度 17 层，其他模块固定为基础模型。MLP 和中间层的敏感性呈倒 U 型；RL 在底部和顶部低敏感区域仍有较大更新范数，OPD 更倾向于把预算集中到中间层 MLP。

这里的证据是“局部替换后的全局性能响应”，可用于定位功能相关区域，但不能直接推出模块之间不存在交互，也不能把干预敏感度等同于严格的因果贡献。

### 2.3 性质 2：早期低秩锁定

论文逐个分析每个 MLP/Attention 权重矩阵的更新 $\Delta W$，而不是直接计算完整参数空间的 Hessian。若

$$
\Delta W=U\Sigma V^\top,
$$

作者报告四类指标：

- 谱范数：$\sigma_1$；
- 谱范数/Frobenius 范数比：
    $$
    \rho=\frac{\sigma_1}{\sqrt{\sum_j\sigma_j^2}};
    $$
- 有效秩（谱熵秩）：
    $$
    \operatorname{rank}_{\mathrm{eff}}=\exp\left(-\sum_i\bar\sigma_i\log\bar\sigma_i\right),
    \qquad \bar\sigma_i=\frac{\sigma_i}{\sum_j\sigma_j};
    $$
- 前 1% 子空间范数比：令 $k=\lceil r/100\rceil$，
    $$
    R_{\mathrm{Top-1\%}}=
    \sqrt{\frac{\sum_{i=1}^k\sigma_i^2}{\sum_{j=1}^r\sigma_j^2}}.
    $$

OPD 的典型结果是更大的谱集中度、更低的有效秩和更高的 Top-1% 能量占比。例如 8B 汇总值中，谱范数/Frobenius 比为 36.8% 对 32.7%，有效秩为 2341 对 2754，Top-1% 比为 94.7% 对 88.5%（OPD 对 RL）。这些是每个矩阵的平均统计，矩阵尺寸不同，因此跨模块直接平均需要谨慎解释。

作者又将谱分为 Top-$k$% 和 Bottom-$k$%：Top 部分先做等 Frobenius 范数缩放再比较方向质量；Bottom 部分不做缩放，观察原始训练状态下的边际贡献。前 10% 秩通常能恢复超过 95% 的全模型推理收益，而尾部需要更大的秩预算且收益很小；RL 的尾部范数约为 OPD 的 1.6--2.5 倍，但性能回报有限。

### 2.4 方向稳定性、轨迹和外推

主子空间分析包括：

- 中间 checkpoint 与最终 checkpoint 的 Top-$k$ 子空间余弦相似度，$k=1,\ldots,20$；
- 秩 -1 子空间 $\mathcal{U}_1$ 的 t-SNE 轨迹（定性）；
- 对 checkpoint 轨迹做 PCA，计算前两个主成分的解释方差比：
    $$
    \mathrm{EVR}_{0:2}=\frac{\lambda_1+\lambda_2}{\sum_i\lambda_i}.
    $$

OPD 在早期阶段与最终子空间的余弦相似度更高，中间层最高约 0.9；PCA 的二维解释方差比也更高。t-SNE 只能说明可视化中的局部聚集，真正的定量证据应看余弦相似度和 PCA explained variance。

正文中的干预显示，10% 训练进度的 OPD checkpoint 在把各模块 Frobenius 范数缩放到最终值后，可恢复最终推理性能约 80%。附录将缩放系数记为 $\beta$；$\beta\approx0.8$--$1.2$ 通常较好，过大时 KL 和准确率反弹。这是“方向已经形成、幅值尚未长足”的证据，但因为缩放使用了最终 checkpoint 的范数，它是后验诊断，不是严格的在线算法。

### 2.5 EffOPD

在 $t=2^n$ 的 checkpoint 触发外推。对 $n\geq1$，用相邻指数 checkpoint 的位移

$$
\Delta_n=W_{2^n}-W_{2^{n-1}}
$$

作为局部方向，构造

$$
\widetilde W_{n,k}=W_{2^n}+2k\Delta_n,\qquad k=1,\ldots,5.
$$

每个候选在 50 个验证样本上评估，性能不再提升就停止外推，否则接受当前候选。最差情况下第一个候选失败，退化为普通 OPD。论文在数学和代码任务上报告平均约 $3\times$ 的训练加速，并与固定外推的 AlphaOPD/ExOPD 比较了自适应验证的好处。

## 3. F.5：从 logits 到局部谱动力学

### 3.1 第一步：logit 线性化

对 token 上下文 $c=(x,y_{<t})$，设学生 logits 为 $z_\theta(c)$，教师 logits 为 $z^\star(c)$，基础参数为 $\theta_0$，参数位移为 $\Delta\theta=\theta-\theta_0$。定义

$$
J_c=\left.\frac{\partial z_\theta(c)}{\partial\theta}\right|_{\theta=\theta_0},
\qquad
r_c=z^\star(c)-z_0(c).
$$

在基础模型邻域内

$$
z_\theta(c)\approx z_0(c)+J_c\Delta\theta,
\qquad
z_\theta(c)-z^\star(c)\approx J_c\Delta\theta-r_c.
$$

这一步把神经网络的非线性函数学习问题变成了输出空间残差经过 Jacobian 拉回参数空间的问题。

### 3.2 第二步：token 输出空间的 Fisher

令 $p_z=\operatorname{softmax}(z)$。固定上下文时，反向 KL 在教师分布附近的二阶项为

$$
F_c^\star=\operatorname{Diag}(p^\star)-p^\star {p^\star}^{\top}.
$$

论文为了围绕基础模型线性化，将它替换为

$$
F_c=\operatorname{Diag}(p_0)-p_0p_0^\top.
$$

这里的 $F_c$ 是词表输出分布上的 categorical Fisher；它不是参数空间完整 Fisher，也不是训练损失的完整 Hessian。若 $p_0$ 与 $p^\star$ 差距很大，替换会带来额外近似误差。

### 3.3 第三步：局部二次目标

将上述两步合并，得到

$$
\mathcal L_{\mathrm{OPD}}(\Delta\theta)
\approx
\frac12\mathbb E_c\left[(J_c\Delta\theta-r_c)^\top F_c(J_c\Delta\theta-r_c)\right].
$$

展开后

$$
\mathcal L_{\mathrm{OPD}}(\Delta\theta)
=\frac12\Delta\theta^\top A\Delta\theta-b^\top\Delta\theta+\mathrm{const},
$$

其中

$$
A=\mathbb E_c[J_c^\top F_cJ_c],
\qquad
b=\mathbb E_c[J_c^\top F_cr_c].
$$

因此

$$
g(\Delta\theta)=A\Delta\theta-b.
$$

$A$ 是 $J_c$ 在 Fisher 度量下的 pullback，也可看成输出空间 Gauss--Newton 矩阵；它一般不等于完整参数 Hessian。若把 softmax/交叉熵的模型曲率、残差二阶项和 on-policy 分布导数全部保留，真实 Hessian 会包含更多项。

更一般地，若损失写成 $\ell(f_\theta(x))$，其参数 Hessian 可分解为

$$
H_\theta
=J_\theta^\top H_{\ell}J_\theta
+\sum_a\frac{\partial\ell}{\partial f_a}\nabla_\theta^2 f_a.
$$

第一项是 generalized Gauss--Newton；在交叉熵/分布匹配问题中，$H_{\ell}$ 常被 Fisher 或 Fisher-like 矩阵近似。第二项是模型输出对参数的二阶导数，非线性网络中并不自动为零。F.5 实际保留的是第一项的局部近似，因此最好把 $A$ 叫作 pullback Fisher 或 Gauss--Newton-like curvature，而不是完整 Hessian。

还要区分三种常被混称的 Fisher：true Fisher（对模型自身分布取期望的 score 外积）、empirical/observed Fisher（用标签或采样标签的梯度外积），以及 F.5 的输出 Fisher $\operatorname{Diag}(p)-pp^\top$ 再经 $J_c^\top(\cdot)J_c$ 拉回参数空间。categorical Fisher 对全 1 的 logit 平移方向有零特征值，因此 $A$ 可能退化；这也是文中使用 $A^\dagger$ 的原因。

### 3.4 第四步：固定步长梯度下降的谱响应

从 $\Delta\theta_0=0$ 出发，固定步长梯度下降满足

$$
\Delta\theta_{s+1}=(I-\eta A)\Delta\theta_s+\eta b,
$$

并有

$$
\Delta\theta_s=\left[I-(I-\eta A)^s\right]A^\dagger b.
$$

这里隐含了稳定步长条件 $0<\eta<2/\lambda_{\max}(A)$（若只讨论单调的逐方向响应，通常还需更强的 $\eta\lambda_i\leq1$）。当步长接近稳定边界或使用 Adam、clip、momentum 时，闭式谱响应不再直接描述实际参数轨迹。

令 $A=U\Lambda U^\top$，$b=U\beta$，则

$$
\Delta\theta_s=
\sum_{i:\lambda_i>0}
\frac{1-(1-\eta\lambda_i)^s}{\lambda_i}\,\beta_i u_i.
$$

解释是：每个方向的最终幅值由 $\beta_i/\lambda_i$ 决定，有限步的响应由 $1-(1-\eta\lambda_i)^s$ 决定。大曲率方向通常更快响应；若 $b$ 主要投影在前 $k$ 个方向且存在谱隙 $\lambda_k\gg\lambda_{k+1}$，早期更新就会集中在低维子空间。

论文把这一点写成尾部界：若

$$
\Vert P_{U_k^\perp}b\Vert\leq\epsilon\Vert b\Vert,
\qquad \epsilon\ll1,
$$

则有限步尾部贡献可由

$$
\rho_\perp(s)=\max_{i>k,\lambda_i>0}
\frac{\left\vert1-(1-\eta\lambda_i)^s\right\vert}{\lambda_i}
$$

控制，进而得到 $\Vert\text{tail}\Vert\leq\rho_\perp(s)\epsilon\Vert b\Vert$。这确实是一个清楚的局部充分条件，但关键的 $b$ 集中性和谱隙不是由论文从 LLM 训练目标中严格推出的。

### 3.5 第五步：模块抑制

把参数和 Jacobian 按模块拆分：$\Delta\theta=(\Delta\theta_1,\ldots,\Delta\theta_M)$、$J_c=[J_{c,1},\ldots,J_{c,M}]$。定义

$$
b_m=\mathbb E_c[J_{c,m}^\top F_cr_c],
\qquad
A_{mn}=\mathbb E_c[J_{c,m}^\top F_cJ_{c,n}].
$$

局部最优点满足

$$
\sum_nA_{mn}\Delta\theta_n^\star=b_m.
$$

如果跨模块耦合 $A_{mn}$ 足够小且 $b_m\approx0$，则

$$
\Delta\theta_m^\star\approx A_{mm}^{-1}b_m\approx0.
$$

这给“embedding、底层、顶层模块更新较小”的解释，但需要强调：真正的 Transformer 模块有强交互，$A_{mn}$ 是否小需要实测，不能仅凭干预曲线假定。

### 3.6 第六步：与 RL 梯度的比较

论文将单 token 的 RL 信号写成

$$
g_{\mathrm{RL}}\approx\mathbb E\left[A_tJ_{c_t}^\top(e_{y_t}-p_0(c_t))\right],
$$

而 OPD 在局部近似初始点的驱动项为 $b=\mathbb E[J_c^\top F_cr_c]$。直觉是：OPD 有 token 级教师残差，RL 依赖标量 advantage；后者在稀疏奖励时方差和信用分配困难更明显。

但文中的协方差结论

$$
\operatorname{Tr}(\Sigma_{\mathrm{RL}})>\operatorname{Tr}(\Sigma_{\mathrm{OPD}})
$$

只是带有独立性、单 token 简化和稀疏奖励前提的经验预期，不是无条件不等式。$F_c$ 会压缩输出方向，RL 与 OPD 的样本分布、优势估计、时间相关性和 batch 处理也不同，严谨比较需要在同一采样协议下估计完整梯度协方差。

## 4. “论文证明了什么”与“论文启发式解释了什么”

### 4.1 直接由实验支持的内容

- 在论文的模型、数据和训练配方上，OPD 的每范数推理收益通常高于 RL。
- OPD 的权重更新矩阵在谱范数/Frobenius 比、有效秩和 Top-1% 能量比上更集中。
- OPD 的早期 Top-$k$ 子空间与最终子空间余弦相似度更高。
- 局部 MLP/Attention 注入显示中间层 MLP 更敏感，而 RL 在外围层有更多更新范数。
- 早期 OPD checkpoint 做模块级后验范数缩放后可恢复相当部分性能。

### 4.2 不能过度外推的内容

- 每个矩阵的 SVD 低秩，不等于全参数空间 $A$ 的前 $k$ 个特征方向；二者缺少严格等价桥梁。
- $b$ 的“低秩”主要是对 token residual 稀疏/低维的解释性假设，论文没有从教师训练机制证明它必然成立。
- 局部二次目标不包含完整 on-policy 分布演化；论文在实验设置中也明确把 OPD 的完整未来项截断成即时 token 近似。
- $\beta$ 范数缩放使用最终 checkpoint 范数，不能直接理解为可在线获得的 oracle-free 方法。
- t-SNE 是可视化，不应作为方向稳定性的主要定量证据。
- 论文 checklist 自己承认未提供所有实验的误差条/显著性检验，计算资源报告也不完整。

另外，F.5 虽然在 checklist 中被称为“理论分析”，源文并没有给出带编号的定理、引理和完整证明；它更准确地说是局部近似推导加上一个带条件的谱尾部界。阅读时不要把“充分条件 + 机制直觉”误读成已经证明了真实 OPD 训练一定产生早期低秩锁定。

## 5. 论文直接引用的内容：按作用归类

### 5.1 OPD、RL 和后训练

| 文献                                                                                                                                       | 论文中承担的作用                                     | 与本研究的关系                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------- |
| Agarwal et al., 2024, [On-policy distillation of language models: Learning from self-generated mistakes](https://arxiv.org/abs/2306.13649) | GKD/OPD 的基本范式：学生在自生成轨迹上接受教师反馈   | F.5 的同策略上下文来源                                        |
| Yang et al., 2025, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)                                                              | 说明 OPD/RLVR 后训练在推理模型中的工程背景           | 实验模型和训练动机                                            |
| Cai et al., 2025, [On Predictability of RL Dynamics for LLMs](https://arxiv.org/abs/2510.00553)                                            | RL 的秩 -1 主导、秩 -1 线性动态与 AlphaRL            | 本文把类似谱诊断迁移到 OPD，并作 RL/OPD 对比                  |
| Yang et al., 2026, [Learning beyond Teacher](https://arxiv.org/abs/2602.12125)                                                             | reward extrapolation、token 级蒸馏与 RLVR 的统一背景 | 本文的 OPD 配方和实验基线                                     |
| Li et al., 2026, [Rethinking OPD](https://arxiv.org/abs/2604.13016)                                                                        | 学生/教师思维模式兼容性与真实新能力条件              | 说明 F.5 的 $r_c$ 不是任意教师残差                            |
| Fu et al., 2026, [Revisiting OPD](https://arxiv.org/abs/2603.25562)                                                                        | token 级 OPD 的偏差、方差界和失败模式                | 对本文“即时 token 近似”和长轨迹稳定性的关键补充               |
| Song & Zheng, 2026, [A Survey of OPD](https://arxiv.org/abs/2604.00626)                                                                    | 用同策略样本统一 $f$-散度视角整理 OPD                | 适合作为 OPD 总览入口                                         |
| Xu et al., 2026, TIP: Token Importance in OPD                                                                                              | token 重要性和教师残差支撑集的直接相关工作           | 对 F.5 中“关键 token 导致 $b$ 集中”的假设提供潜在实证检验方向 |

### 5.2 机制定位与模块解释

| 文献                                                                                                          | 工具/贡献                                                   |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Geva et al., 2021, [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) | 将 FFN/MLP 解释为 key-value memory，支撑中间 MLP 的功能解释 |
| Meng et al., 2023, [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)       | 局部层块干预和知识定位范式                                  |
| Vig et al., 2020, _Investigating Gender Bias in Language Models Using Causal Mediation_                       | 机制干预/因果中介的先例                                     |
| Cai et al., 2024, [Locating and Mitigating Gender Bias](https://arxiv.org/abs/2403.14409)                     | Transformer 模块和滑动窗口定位的近邻工作                    |

### 5.3 SVD、低秩和谱度量

| 文献                                                                                                                 | 论文使用方式                               | 需要注意的地方                                                   |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| Zhang, 2015, [The Singular Value Decomposition, Applications and Beyond](https://arxiv.org/abs/1510.08532)           | SVD 背景引用                               | 主要是数学/应用综述                                              |
| Koren et al., 2009, [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263)   | 当前 `.bib` 中的 `5197422` 被用作 SVD 引用 | 该条目实际是推荐系统矩阵分解，不是 SVD 基础来源                  |
| Eckart & Young, 1936, [The Approximation of One Matrix by Another of Lower Rank](https://doi.org/10.1007/BF02288367) | 低秩近似；论文也把它用于 PCA 说明          | 更准确地说是截断 SVD/最佳低秩近似经典来源，不能替代 PCA 专门引用 |
| Roy & Vetterli, 2007, [The Effective Rank](https://infoscience.epfl.ch/handle/20.500.14299/10320)                    | 有效秩/谱熵秩                              | 与论文的 entropy-rank 指标直接对应                               |
| Mathias, 1990；Al-Natoor, 2024                                                                                       | 谱范数和范数不等式                         | 与 LLM 机制本身距离较远，属于指标定义支撑                        |

### 5.4 训练设置和资源

论文还引用了 PPO、DAPO、DeepMath-103K、MATH-12K、Eurus-RL-Code、verl/HybridFlow、Qwen 技术报告以及 Codeforces/Taco/AIME/MINERVA/GPQA 等基准。这些条目用于复现实验，不是 F.5 理论工具。

引用完整性方面，正文使用 `OpenAI2025`，但 `.bib` 的 key 是小写 `openai2025`；这会造成 BibTeX 引用缺失。另有若干 2026 条目是 arXiv 或技术报告，应在正式版本中注明版本和状态。

## 6. LLM 时代的工具谱系

可以把相关工作按“测量对象”分成五层：

1. **函数空间局部几何**：Jacobian、NTK、Gauss--Newton、Fisher pullback。回答“哪些参数方向最能改变 logits/函数”。
2. **损失曲率**：Hessian 最大特征值、迹、HVP、sharpness、K-FAC/块 Kronecker 近似。回答“哪些方向对 loss 最敏感、步长是否稳定”。
3. **更新/梯度子空间**：SVD、PCA、随机 SVD、online PCA、principal angle、cosine。回答“实际训练走了哪些低维方向”。
4. **模块功能与因果干预**：embedding replacement、layer-window injection、activation patching、causal mediation。回答“更新落在哪些功能区域”。
5. **压缩和训练控制**：低秩投影、Fisher/Hessian 加权 SVD、自然梯度、rank scheduling、量化/剪枝、方向外推。回答“如何把诊断转成算法”。

其中最容易混淆的是第 1--3 层：F.5 的 $A=J^\top FJ$ 是函数空间 Fisher 度量拉回参数空间；主文的 $\Delta W$ SVD 是单个权重矩阵的经验更新谱；Hessian 的 eigenspace 则是 loss 曲率谱。三者可能相关，但没有自动等价。

## 6.1 一条较完整的历史脉络

| 时期       | 代表工作                                                            | 工具与核心问题                                                        | 对本文的启发                                       |
| ---------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------- |
| 1936       | Eckart--Young                                                       | 最佳低秩矩阵近似、截断 SVD                                            | 为什么 Top-$k$ 奇异方向是固定范数下的最佳矩阵近似  |
| 1989--1993 | Optimal Brain Damage；Optimal Brain Surgeon                         | Hessian 对角/逆 Hessian 的剪枝敏感度                                  | “二阶敏感度可以决定保留哪些参数”                   |
| 1998       | Amari, Natural Gradient                                             | Fisher-Rao 度量、自然梯度                                             | F.5 中 $F$ 的信息几何来源                          |
| 2015       | Martens--Grosse, K-FAC                                              | Kronecker 因子化 Fisher/Hessian 近似                                  | 把不可存储的二阶矩阵变成可计算预条件器             |
| 2018       | Jacot et al., NTK                                                   | Jacobian kernel、线性化训练                                           | F.5 的 $J_c$ 与谱动力学的理论邻居                  |
| 2019--2021 | HAWQ/HAWQ-V2；WoodFisher                                            | Hessian trace、块/逆 Fisher 的混合精度与压缩                          | 曲率可用于量化/剪枝，而不只是解释训练              |
| 2022--2023 | LoRA；AdaLoRA；GPTQ；SparseGPT                                      | 低秩适配、动态 rank、二阶量化/剪枝                                    | 低秩限制与曲率感知压缩进入 LLM                     |
| 2023--2024 | KFT、SVD-LLM、GaLore、PiSSA、OSD、SVFT                              | NTK fine-tuning、权重/梯度 SVD、在线子空间                            | 分别研究函数核、更新谱、权重谱和优化状态谱         |
| 2025       | Linearization、AdaRankGrad、SUMO、WeLore、QERA、RL Predictability   | NTK 谱、梯度 rank、动态 SVD、Hessian/Fisher、RL 轨迹                  | 将谱诊断扩展到全参数训练和推理后训练               |
| 2026       | FisherAdapTune、CG-LoRA、GRIT、GFWSVD、critical sharpness、本文 OPD | Fisher 漂移、曲率白化、K-FAC、跨层 Fisher、方向性 sharpness、OPD 外推 | 从“观察低秩/曲率”转向在线选择参数、rank 和训练步长 |

这条脉络的主线是：先用 Hessian/Fisher 衡量敏感度，再用 Jacobian/NTK 把参数变化映射到函数变化，最后用 SVD/子空间方法压缩或控制真正发生的更新。本文处在“LLM 后训练动力学诊断”这一交叉点上。

## 7. 2024--2026 代表性论文与 F.5 的对应关系

### 7.1 直接研究线性化、NTK、局部几何

| 论文                                                                                                                           | 状态      | 工具                                        | 关键结论/阅读重点                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------ | --------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [Linearization Explains Fine-Tuning in LLMs](https://openreview.net/forum?id=tdwRIP6NG2), NeurIPS 2025                         | 已接收    | 参数 $\ell_2$ 归纳偏置、正定 NTK、谱扰动界  | 与 F.5 最直接的理论邻居：分析何时完整微调接近线性化微调，以及选层如何改变 NTK 谱；可用来补足 F.5 的 Jacobian 视角。 |
| [A Kernel-Based View of Language Model Fine-Tuning](https://openreview.net/forum?id=49dTFIGdx8), ICML 2023                     | 已接收    | kernel/NTK                                  | 先读它再读 2025 线性化论文，建立“函数变化由核谱控制”的基础。                                                        |
| [Understanding Linear Probing then Fine-tuning from NTK Perspective](https://openreview.net/forum?id=1v4gKsyGfe), NeurIPS 2024 | 已接收    | NTK、层选择                                 | 连接 probing 和 fine-tuning 的阶段性动力学。                                                                        |
| [Curvature-Guided LoRA](https://arxiv.org/abs/2603.29824), 2026                                                                | arXiv，新 | 函数空间 prediction alignment、曲率白化梯度 | 把“匹配 full fine-tuning 的函数输出”写成二阶问题，和 F.5 的 $J^\top FJ$ 最接近的算法化延伸之一。                    |

### 7.2 直接研究梯度/更新低秩与子空间动态

| 论文                                                                                                                                | 状态                             | 工具                                                | 关键结论/与本文的关系                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [GaLore](https://openreview.net/forum?id=hYHsrKDiX7), ICML 2024 oral                                                                | 已接收                           | 梯度 SVD 投影、动态子空间                           | 允许全参数学习但压缩 optimizer state；是 LLM 梯度低秩工程化的关键基线。                                                                          |
| [Memory-Efficient LLM Training with Online Subspace Descent](https://openreview.net/forum?id=P8rTCT6g45), NeurIPS 2024              | 已接收                           | online PCA、子空间下降、收敛保证                    | 用 online PCA 替代每步 SVD，直接关联“子空间如何更新”的问题。                                                                                     |
| [PiSSA](https://openreview.net/forum?id=6ZBHIEtdP4), NeurIPS 2024 spotlight                                                         | 已接收；OpenReview 缓存引文数 35 | 权重 SVD、主奇异分量初始化                          | 把预训练权重的 principal singular components 分给 adapter；与本文“更新主子空间”不同，不能混为更新谱分析。                                        |
| [SVFT](https://openreview.net/forum?id=uS0PwIBzC0), NeurIPS 2024                                                                    | 已接收                           | 固定权重奇异向量、稀疏系数矩阵                      | 只训练奇异向量外积的系数，适合比较“权重谱基”与“更新谱基”。                                                                                       |
| [AdaRankGrad](https://openreview.net/forum?id=LvNROciCne), ICLR 2025                                                                | 已接收                           | online randomized SVD、梯度 rank scheduling         | 形式化声称 layer gradient rank 随训练趋向 rank-1；是“早期低秩锁定”最直接的优化器邻居。                                                           |
| [Does SGD really happen in tiny subspaces?](https://openreview.net/forum?id=v6iLQBoIJw), ICLR 2025                                  | 已接收                           | Hessian dominant subspace 投影/消融                 | 重要反例：梯度看似对齐 Hessian 顶部子空间，但只保留该子空间反而不能继续降 loss，投影掉它也可能同样有效；提醒不要把谱对齐直接解释成因果有效方向。 |
| [Breaking the Frozen Subspace](https://openreview.net/forum?id=ZdmmOAN4h3), NeurIPS 2025                                            | 已接收                           | dominant subspace、importance sampling              | 发现 dominant subspace 在预训练中会冻结，提出随机重要性采样打破固定子空间；对 EffOPD 的“沿早期方向外推”是重要反向压力测试。                      |
| [SUMO](https://openreview.net/forum?id=DIjRvEKOeG), NeurIPS 2025                                                                    | 已接收                           | 动态子空间内精确 SVD、moment orthogonalization      | 以精确 SVD 替代 Newton--Schulz 近似，利用各向异性改善收敛；是“谱诊断转优化器”的代表。                                                            |
| [From Low Rank Gradient Subspace Stabilization to Low-Rank Weights / WeLore](https://openreview.net/forum?id=fsf7LhbYdf), ICML 2025 | 已接收                           | Hessian 分析、梯度子空间稳定、非均匀权重 rank       | 认为梯度子空间稳定性与后续权重低秩表达有关，并据此做压缩和 PEFT；可用于检验 F.5 的“谱结构跨时间稳定”是否具有层间差异。                           |
| [Reinforcement Learning Finetunes Small Subnetworks](https://openreview.net/forum?id=0NdS4xCngO), NeurIPS 2025                      | 已接收                           | 参数更新稀疏性、子网络重训                          | 发现 RL 只需更新约 5%--30% 参数即可恢复性能，但这些矩阵内的更新仍可近乎 full-rank；这是本文“RL 更新更弥散”的重要互补/潜在张力。                  |
| [Low Rank Gradients and Where to Find Them](https://openreview.net/forum?id=de3kwOXQ9e), NeurIPS 2025                               | 已接收                           | spiked data、mean-field/NTK、rank-one gradient 分解 | 在两层网络中给出低秩梯度来源的理论，能帮助理解“残差和数据结构如何产生低秩驱动”，但尚未直接覆盖 Transformer/OPD。                                 |

### 7.3 Fisher/Hessian/自然梯度用于训练、PEFT 与压缩

| 论文                                                                                     | 状态      | 工具                                                             | 关键结论/与 F.5 的关系                                                                                               |
| ---------------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [CAMEx](https://openreview.net/forum?id=nT2u0M0nf8), ICLR 2025                           | 已接收    | Fisher 近似、natural gradient、曲率感知 expert merging           | 将合并从欧氏参数空间搬到非欧氏几何；可作为 F.5 中 Fisher-Rao 解释的工程实例。                                        |
| [AdaFisher](https://openreview.net/forum?id=puTxuiK2qO), ICLR 2025                       | 已接收    | diagonal block-Kronecker Fisher                                  | 用块 Kronecker Fisher 做二阶预条件；适合学习 K-FAC/Fisher 近似如何在大模型上落地。                                   |
| [HiZOO](https://openreview.net/forum?id=bEqI61iBue), ICLR 2025                           | 已接收    | diagonal Hessian、zeroth-order optimizer                         | 将 Hessian 对角线引入 ZO fine-tuning，在 350M--66B 模型上改善稳定性；说明二阶信息不仅用于解释，也能直接控制训练。    |
| [QERA](https://openreview.net/forum?id=LB5cKhgOTu), ICLR 2025                            | 已接收    | SVD、量化误差、激活/输出误差解析重构                             | 将低秩误差补偿从权重 Frobenius 误差推广到输出误差，连接 SVD 与任务相关几何。                                         |
| [Generalized Fisher-Weighted SVD](https://arxiv.org/abs/2505.17974), 2025                | arXiv     | observed Fisher、Kronecker factor、Fisher-weighted SVD           | 处理 Fisher 的非对角相关，优于仅对角 Fisher 的压缩；与 F.5 的 $F$ 加权更新有直接方法论联系。                         |
| [Scalable Kronecker-Fisher Approximation](https://arxiv.org/abs/2609.02451), 2026        | arXiv，新 | 跨层 Kronecker-Fisher、Hessian 分析                              | 面向 billion-parameter 模型分析脆弱层和跨层相关，目标是量化、稀疏化和 adaptive low-rank decomposition。              |
| [HESTIA](https://arxiv.org/abs/2601.20745), 2026                                         | arXiv     | Hessian trace、soft quantization、温度退火                       | 用 tensor-wise Hessian trace 控制极低比特 QAT 的离散化速度；是曲率诊断驱动训练日程的实例。                           |
| [A Scalable Measure of Loss Landscape Curvature](https://arxiv.org/abs/2601.16979), 2026 | arXiv     | critical sharpness、方向性曲率、Edge of Stability                | 不直接求最大 Hessian 特征值，仅用少量前向评估 update-direction curvature，适合将“曲率随训练如何演化”扩展到 7B 规模。 |
| [GRIT](https://arxiv.org/abs/2601.00231), 2026                                           | arXiv     | K-FAC preconditioning、Fisher eigenspace reprojection、动态 rank | 把自然梯度代理、Fisher 重投影和 rank adaptation 合并到 LoRA。                                                        |
| [Fisher-Guided Progressive Parameter Selection](https://arxiv.org/abs/2606.10196), 2026  | arXiv     | Fisher distribution drift、Jensen--Shannon 距离、PAC-Bayes       | 依据 Fisher 几何随时间的漂移逐步冻结参数组；和本文“功能冗余抑制/早期稳定”高度相近，但使用了更明确的冻结准则。        |

### 7.4 SVD、谱几何与 LLM 结构分析

| 论文 | 状态 | 关键工具/发现 |
| ---------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| [Tracing the Representation Geometry of LMs](https://openreview.net/forum?id=FDruZlKWUb), NeurIPS 2025 | 已接收 | RankMe 有效秩、谱衰减；发现预训练和后训练存在 warmup、entropy-seeking、compression-seeking 几何阶段，RLVR 更偏 compression-seeking。 |
| [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://openreview.net/forum?id=xp7B8rkh7L), NeurIPS 2025 | 已接收 | 权重 SVD、intruder dimensions、奇异值因果干预 | 说明 LoRA 与 full fine-tuning 的谱结构并不等价，并将遗忘定位到新奇异方向；是本文 SVD 干预设计的强相关邻居。 |
| [KaSA](https://arxiv.org/abs/2412.06071), ICLR 2025 | 已接收/ arXiv | knowledge-aware singular values | 用任务相关知识激活奇异值，连接 SVD 和选择性适配。 |
| [EVA](https://openreview.net/forum?id=movsqor65f), NeurIPS 2025 | 已接收 | activation covariance 的增量 SVD、explained variance、自适应 rank | 选择最大激活方差方向以最大化期望梯度信号；与本文“更新能量集中”互补，但对象是初始化时的激活几何。 |
| [FedSVD](https://openreview.net/forum?id=Qq19n9LZ97), NeurIPS 2025 | 已接收 | 聚合更新 SVD、正交重参数化 | 在 DP 联邦 LoRA 中用 SVD 重构主方向，说明谱重参数化也能缓解噪声放大。 |

### 7.5 2026 年值得优先追踪的新稿

| 论文                                                                                                 | 工具                               | 为什么与 F.5 相关                                                                        |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| [Gradient Intrinsic Dimensionality Alignment](https://openreview.net/forum?id=kObvnQ6pUx), ICLR 2026 | intrinsic dimension、低秩/全参对齐 | 直接讨论低秩适配和 full fine-tuning 的有效维度差距，适合检验“更新低秩是否等于函数低维”。 |
| [On Predictability of RL Dynamics for LLMs](https://openreview.net/forum?id=SdHmA6BYVJ), ICLR 2026   | rank-1 dominance、线性子空间预测   | 与本文的 OPD/RL 方向稳定性比较最接近；其 AlphaRL 可作为 EffOPD 的对照。                  |
| [Curvature-Guided Policy Optimization](https://openreview.net/forum?id=R2EZtdHWJT), ICLR 2026        | policy curvature、二阶 RL 更新     | 将曲率工具从 PEFT/压缩带到 LLM 推理 RL，适合比较 RL 的 $A_t$ 信号和 OPD 的 $r_c$ 信号。  |
| [Three Forward, One Backward](https://openreview.net/forum?id=373rsDQsq4), ICLR 2026                 | extra forward、full-rank update    | 从计算预算角度反驳“必须低秩才高效”的默认前提，可作为低秩训练的效率基线。                 |
| [Fisher-Guided Progressive Parameter Selection](https://arxiv.org/abs/2606.10196), 2026              | Fisher drift、JS 距离、渐进冻结    | 与本文功能冗余规避相似，但把“低收益模块”转化为可在线冻结准则。                           |
| [Scalable Kronecker-Fisher Approximation](https://arxiv.org/abs/2609.02451), 2026                    | 跨层 Kronecker-Fisher              | 直接提供检验 $A_{mn}$ 跨模块耦合是否可忽略的工具。                                       |

## 8. 影响力与新颖性怎么取舍

- **影响力/基础必读**：Amari 的 natural gradient、LeCun 的 Optimal Brain Damage、Hassibi 的 Optimal Brain Surgeon、Jacot 等的 NTK、Martens--Grosse 的 K-FAC、LoRA、GaLore、PiSSA、QERA。
- **与本文 F.5 最直接**：Linearization Explains Fine-Tuning、A Kernel-Based View of Fine-Tuning、AdaRankGrad、WeLore、CAMEx、Generalized Fisher-Weighted SVD。
- **对本文结论最重要的反例/校验**：Does SGD really happen in tiny subspaces、Breaking the Frozen Subspace、RL Finetunes Small Subnetworks、LoRA vs Full Fine-tuning。
- **最新但尚未形成稳定共识**：Scalable Kronecker-Fisher、Curvature-Guided LoRA、HESTIA、FisherAdapTune、GRIT、C-PTQ、F-WANDA。它们适合追踪方向，不应和已充分验证的经典结论放在同一证据等级。

## 9. 建议的学习路线

### 阶段 A：数学和几何基础

1. 线性代数：SVD、Eckart--Young、谱范数、Frobenius 范数、有效秩、主角/子空间距离。
2. 信息几何：categorical Fisher、自然梯度、KL 的二阶展开、Gauss--Newton、K-FAC。
3. 核方法：Jacobian、NTK、线性化训练、kernel eigenspectrum。
4. 优化动力学：固定步长二次问题、谱半径、Hessian top eigenspace、Edge of Stability。

### 阶段 B：读论文

推荐顺序：

1. Agarwal 2024 GKD，理解 on-policy distillation 的采样和目标。
2. LoRA 与 GaLore，理解低秩参数化和低秩梯度投影的差别。
3. PiSSA、SVFT、EVA、QERA，比较“权重 SVD、奇异向量基、激活 SVD、误差 SVD”。
4. Online Subspace Descent、AdaRankGrad、SUMO，理解动态子空间如何进入优化器。
5. Linearization Explains Fine-Tuning、KFT/NTK 论文，回到 F.5 的 Jacobian--核视角。
6. CAMEx、AdaFisher、HiZOO、GFWSVD，学习 Fisher/Hessian 的近似和计算取舍。
7. Does SGD really happen in tiny subspaces、Breaking the Frozen Subspace、RL Finetunes Small Subnetworks，主动检查“低秩/主子空间”叙事的边界。
8. 最后读 OPD 2026 系列和本文，判断 token residual、teacher support 和 on-policy drift 如何改变几何结论。

### 阶段 C：最小可复现实验

用一个 1B--7B 开源模型和小型数学数据集，固定同一初始 checkpoint，分别训练 RL/OPD，并保存 0%、10%、30%、50%、100% checkpoint：

1. 对每个矩阵计算 SVD、谱范数/Frobenius 比、有效秩、Top-1% 和 Top-$k$ 能量。
2. 计算每个 checkpoint 到最终 checkpoint 的 principal angle/cosine，而不是只画 t-SNE。
3. 同时用 Hutchinson HVP 估计 loss Hessian trace/top eigenvalue，比较它与更新 SVD 的相关性。
4. 计算输出 Jacobian 的 Fisher pullback 的低维近似，至少用块对角或 K-FAC；检验 $A$ 的主特征子空间是否真的对应每个权重矩阵的 SVD 主方向。
5. 做 embedding replacement、层窗口注入和随机窗口对照，报告多随机种子误差条。
6. 在线保存早期局部方向，比较 EffOPD 外推、固定步长外推和随机子空间采样；禁止使用最终 checkpoint 范数作为在线信息。
7. 对 OPD 额外比较 full reverse-KL、token-level immediate KL、teacher top-$K$ support matching，直接测试 F.5 忽略 on-policy 分布演化的影响。

## 10. 最值得继续追的三个问题

1. **矩阵 SVD 与 pullback-Fisher eigenspace 是否一致？** 这是本文经验谱分析和 F.5 理论之间缺失的桥梁。需要在同一层、同一 batch 上直接估计两种子空间的 principal angles。
2. **OPD 的低秩来自 teacher residual、Jacobian 还是 optimizer dynamics？** 可分别打乱 token residual、固定 Jacobian、替换 Adam/SGD，并测量谱集中度变化。
3. **早期方向稳定何时失效？** 在教师/学生分布严重不匹配、多轮 agent、视觉语言模型、长轨迹和高熵教师上，OPD 的 $r_c$ 支撑、Fisher 度量和子空间谱隙可能同时变化；这正是 2026 OPD 失败模式论文和本文未来工作的交叉点。

## 11. 已生成的检索报告

本次检索使用了 `paper-search` 技能。完整原始报告保留在：

- `paper-search-reports/20260909-013027-093639-fisher-hessian-jacobian-ntk-llm-training-dynamics-fisher-weighte.md`
- `paper-search-reports/20260909-012449-665881-linearization-explains-fine-tuning-in-large-language-models-trac.md`
- `paper-search-reports/20260909-011519-742053-low-rank-optimization-trajectories-large-language-model-reinforc.md`
- `paper-search-reports/20260909-011644-801423-hessian-fisher-jacobian-ntk-curvature-llm-fine-tuning-natural-gr.md`
- `paper-search-reports/20260909-011644-883797-svd-llm-compression-fisher-weighted-svd-language-model-hessian-g.md`
- `paper-search-reports/20260909-011646-002402-low-rank-gradient-gradient-subspace-spectral-geometry-language-m.md`
- `paper-search-reports/20260909-013259-958452-optimal-brain-damage-hessian-pruning-optimal-brain-surgeon-hessi.md`

这些报告包含每个来源的完整命中、OpenReview URL、年份、venue、摘要和检索诊断；本笔记只抽取与本文工具链最相关的条目。
