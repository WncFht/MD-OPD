# LLM 中 Fisher、Hessian 与 SVD/低秩分解：训练分析与辅助训练论文脉络

**检索截止：2026-09-09。** 本报告围绕三个相互联系但不能混为一谈的对象整理文献：

- **Hessian/曲率**：研究损失面的局部几何、训练稳定性、二阶优化，以及剪枝/量化造成的损失增量。
- **Fisher 信息**：把参数空间看成由模型分布诱导的黎曼几何，用于 natural gradient、持续学习中的参数重要性、模型合并和 Fisher 加权分解。
- **SVD/低秩结构**：把权重、激活、梯度或更新限制到低维谱子空间，用于压缩、PEFT、显存优化和秩分配。

检索使用了 paper-search 技能在 arXiv、OpenAlex、OpenReview、Crossref 等来源检索 1989--2026 年文献，并对题名、年份、venue 和链接做了交叉核验。Semantic Scholar 在本轮匿名访问中遇到 HTTP 429，DBLP 返回 JSON 解析错误，因此不把这两个来源的缺失误认为没有相关论文。

## 1. 先给结论

这条研究线可以压缩成一句话：**Hessian 衡量“沿参数扰动走多陡”，Fisher 衡量“改变模型分布有多贵”，SVD 负责把这种代价集中到少数可学习方向。**

最重要的演化有五步：

1. 1989--2015：从 Optimal Brain Damage/Surgeon 的二阶剪枝、Natural Gradient 和 Hessian-free optimization 开始，建立曲率感知的训练和压缩框架。
2. 2015--2021：K-FAC 让 Fisher/二阶预条件可扩展；EWC/结构化 Laplace 把 Fisher 变成持续学习的参数重要性；Hessian 谱分析成为理解尖锐度、边缘稳定性和过参数化的工具。
3. 2019--2023：HAWQ、Q-BERT、GPTQ、OBC、SparseGPT、QuIP 把 Hessian 或其逆近似用于混合精度量化、一次性剪枝和 2-bit LLM 压缩。
4. 2021--2024：LoRA、AdaLoRA、QLoRA、GaLore、PiSSA、EVA、CorDA 把低秩从“压缩后的近似”推进成“训练时的参数化/子空间”。
5. 2024--2026：研究重点转向 activation/data-aware SVD、可微分 SVD、Fisher-weighted SVD、动态秩分配、现代 Transformer 的可扩展曲率近似，以及预训练时的二阶优化器 Sophia/AdaFisher。

## 2. 统一数学视角

设训练目标为 $L(\theta)$，参数扰动为 $\delta$。

### 2.1 Hessian、Taylor 展开与敏感度

$$
L(\theta+\delta) \approx L(\theta)+g^\top\delta+\frac{1}{2}\delta^\top H\delta,
\qquad g=\nabla_\theta L,\quad H=\nabla_\theta^2L.
$$

在收敛点附近 $g$ 较小，$\delta^\top H\delta$ 就是剪枝、量化、低秩替换的二阶损失代价。实际 LLM 中很少显式构造 $H$：常用 Hessian-vector product、对角/块对角近似、Lanczos 谱估计，或用线性层输入协方差作为 Gauss--Newton/输出重构的代理。

Hessian 是对称但通常**不半正定**的；负特征值对应鞍点或负曲率，顶部特征值控制局部稳定性，谱密度和 trace 可用于比较训练阶段与模型层的敏感度。

### 2.2 Fisher 与 natural gradient

对 $p_\theta(y\mid x)$，Fisher 信息为

$$
F(\theta)=\mathbb{E}_{x,y\sim p_\theta}\left[\nabla_\theta\log p_\theta(y\mid x)\nabla_\theta\log p_\theta(y\mid x)^\top\right].
$$

Natural gradient 使用 $F^{-1}g$ 替代 $g$：

$$
\Delta\theta=-\eta F^{-1}g.
$$

在模型分布和正则条件满足时，$F$ 与负对数似然 Hessian 的期望相关，但 **empirical Fisher**（把真实标签样本梯度外积直接平均）一般不等于 Hessian，也不一定给出好的 natural-gradient 预条件。LLM 中最常见的是对角 Fisher、层块 Fisher 或 K-FAC 的 Kronecker 因子。

### 2.3 SVD、加权低秩和 Fisher/SVD 的交汇

普通 SVD 为 $W\approx U_r\Sigma_rV_r^\top$，优化的是未加权 Frobenius 误差。若关心层输出误差，常见目标是

$$
\min_{\operatorname{rank}(\widetilde W)\le r}\|(W-\widetilde W)X\|_F^2,
$$

其中 $X$ 是校准激活；若关心 Fisher/Hessian 度量，则目标近似为

$$
\min_{\operatorname{rank}(\widetilde W)\le r}
\operatorname{vec}(W-\widetilde W)^\top F\operatorname{vec}(W-\widetilde W).
$$

当 $F$ 能写成 Kronecker 形式时，可以白化后做 weighted SVD。这解释了为什么 ASVD、SVD-LLM、CorDA、EVA 和 Fisher-weighted SVD 会把“保留最大奇异值”改成“保留任务/激活/曲率最重要的方向”。

## 3. 时间脉络

| 时期       | 主要问题                       | 代表方法                                                                | 对 LLM 的遗产                         |
| ---------- | ------------------------------ | ----------------------------------------------------------------------- | ------------------------------------- |
| 1989--2000 | 二阶剪枝、参数空间几何         | OBD、OBS、Natural Gradient                                              | 用局部曲率定义参数重要性              |
| 2010--2016 | 可扩展二阶训练                 | Hessian-free、K-FAC、Kronecker Fisher                                   | 用 HVP/块结构替代完整 Hessian         |
| 2015--2021 | 训练动力学和谱诊断             | Hessian eigenspectrum、PyHessian、EWC、HAWQ                             | 谱、trace、Fisher 进入训练分析与压缩  |
| 2022--2023 | LLM 一次性压缩和预训练优化     | GPTQ、OBC、SparseGPT、QuIP、Sophia、QLoRA                               | 曲率感知 PTQ 与二阶预训练成为实用工具 |
| 2023--2026 | 低秩训练、数据感知分解、动态秩 | ASVD、SVD-LLM、PiSSA、EVA、CorDA、GaLore、Dobi-SVD、Fisher-weighted SVD | 从“压缩权重”转向“学习/更新可行子空间” |

## 4. 奠基论文：曲率、Fisher 与二阶优化

### 4.1 二阶剪枝和 Hessian-free

| 论文                                                                                                                                                      | 年份/venue                   | 核心贡献                                                 | 现在怎么看                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------- | --------------------------------------- |
| [Optimal Brain Damage](https://proceedings.neurips.cc/paper_files/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)（LeCun, Denker, Solla） | 1989, NeurIPS                | 用 Hessian 对角近似估计删除权重的损失增量                | 现代 Hessian pruning 的起点             |
| [Optimal Brain Surgeon](https://openalex.org/W2125389748)（Hassibi, Stork）                                                                               | 1992, NIPS workshop/后续版本 | 用逆 Hessian 处理参数间补偿，比 OBD 的独立权重假设更精确 | GPTQ/OBQ/SparseGPT 的思想源头           |
| [Deep learning via Hessian-free optimization](https://openalex.org/W196761320)（Martens）                                                                 | 2010, ICML                   | 用共轭梯度和 Hessian-vector product 做大网络二阶优化     | 说明“不存 Hessian”也能使用曲率          |
| [Training Deep and Recurrent Networks with Hessian-Free Optimization](https://doi.org/10.1007/978-3-642-35289-8_27)                                       | 2012                         | 将 HF 优化推广到深度/循环网络                            | 连接早期语音模型训练与现代二阶 LLM 优化 |
| [Pruning deep neural networks by optimal brain damage](https://doi.org/10.21437/interspeech.2014-281)                                                     | 2014, Interspeech            | 在 DNN 语音模型上重访 OBD                                | 说明曲率剪枝在大模型前已有工程验证      |

### 4.2 Natural gradient、K-FAC 与持续学习

| 论文                                                                                                                                     | 年份/venue               | 核心贡献                                                  | 适用场景                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------- | ---------------------------------------------- |
| [Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746)（Amari）                                    | 1998, Neural Computation | 以 Fisher 度量替代欧氏参数空间，提出 natural gradient     | 理解参数化不变性、预条件与信息几何             |
| [Revisiting Natural Gradient for Deep Networks](https://openalex.org/W2964309400)                                                        | 2014                     | 讨论深网中的 natural-gradient 近似和实现问题              | 连接理论 Fisher 与可实现优化器                 |
| [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://doi.org/10.48550/arxiv.1503.05671)（Martens, Grosse） | 2015, ICML               | K-FAC 用层输入/输出梯度的 Kronecker 因子近似 Fisher       | 大网络二阶预条件的经典基线                     |
| [A Kronecker-factored approximate Fisher matrix for convolution layers](https://doi.org/10.48550/arxiv.1602.01407)                       | 2016                     | 将 K-FAC 扩展到卷积层                                     | 说明结构化 Fisher 如何适配不同算子             |
| [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)（Kirkpatrick et al.）                   | 2017, PNAS               | 用 Fisher 对角估计旧任务参数重要性，加入 EWC 正则         | LLM 持续预训练/连续微调的基础                  |
| [Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting](https://arxiv.org/abs/1805.07810)                      | 2018                     | 用结构化曲率近似替代 EWC 的纯对角 Fisher                  | 连接 Bayesian/Laplace 与 K-FAC                 |
| [Large-Scale Distributed Second-Order Optimization Using K-FAC](https://doi.org/10.1109/cvpr.2019.01264)                                 | 2019, CVPR               | 将 K-FAC 推到分布式大规模训练                             | 证明曲率方法不只停留在小模型                   |
| [Continual Learning with Extended K-FAC](https://doi.org/10.1109/cvpr42600.2020.00902)                                                   | 2020, CVPR               | 用 K-FAC 结构保留更多跨参数相关性                         | EWC 的结构化升级                               |
| [Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures](https://openreview.net/forum?id=Ex3oJEKS53)           | 2023, NeurIPS            | 针对现代架构重新整理 K-FAC 的层级和实现                   | 与 Transformer/LLM 对接的重要近作              |
| [AdaHessian: An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719)                                  | 2021, AAAI               | 用 Hessian 对角/块统计调节每个参数的步长                  | 通用深网二阶优化器；LLM 中通常要用更便宜的近似 |
| [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://doi.org/10.48550/arxiv.2305.14342)        | 2023                     | 用廉价的 Hessian 对角估计、剪裁和归一化更新预训练语言模型 | 目前最应直接对比 Adam 的 LLM 二阶优化基线之一  |
| [A Layer-Wise Natural Gradient Optimizer for Training Deep Neural Networks](https://openreview.net/forum?id=niG3Yyb6oA)                  | 2024, NeurIPS            | 层级 natural-gradient 优化，降低全局 Fisher 的成本        | 新近可扩展二阶优化方向                         |
| [AdaFisher: Adaptive Second Order Optimization via Fisher Information](https://openreview.net/forum?id=puTxuiK2qO)                       | 2025, ICLR               | 自适应 Fisher 二阶优化                                    | 值得关注，但影响力尚在形成                     |

## 5. Hessian 谱：用于分析训练，而不只是做优化

| 论文 | 年份 | 关注点 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------: | ----------------------------------------------------------- | ---------------------------------- |
| [Empirical Analysis of the Hessian of Over-Parametrized Neural Networks](https://doi.org/10.48550/arxiv.1706.04454)（Sagun et al.） | 2017 | 观察过参数化网络 Hessian 的谱结构、尖锐方向和大量近零特征值 |
| [How noise affects the Hessian spectrum in overparameterized neural networks](https://arxiv.org/abs/1910.00195) | 2019 | 分析噪声、过参数化和 Hessian 谱之间的关系 |
| [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://doi.org/10.48550/arxiv.1901.10159)（Yao et al.） | 2019 | 用特征值密度观察训练过程和优化状态 |
| [PyHessian: Neural Networks Through the Lens of the Hessian](https://doi.org/10.1109/bigdata50022.2020.9378171) | 2020, IEEE Big Data | 提供 Hessian trace、top eigenvalue、density 等工具链 |
| [A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization](https://arxiv.org/abs/2012.03801) | 2020 | 将谱统计用于正则化和训练比较 |
| [Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach](https://doi.org/10.48550/arxiv.2006.09092) | 2020 | 用随机矩阵理论解释 batch size、学习率和谱尺度 |
| [On the Power-Law Hessian Spectrums in Deep Learning](https://doi.org/10.48550/arxiv.2201.13011) | 2022 | 研究 Hessian 特征值的幂律尾部和临界训练现象 |
| [Neglected Hessian component explains mysteries in Sharpness regularization](https://doi.org/10.48550/arxiv.2401.10809) | 2024 | 指出只看部分 Hessian/gradient 项可能误读 sharpness 正则化 |
| [Linearization Explains Fine-Tuning in Large Language Models](https://openreview.net/forum?id=tdwRIP6NG2) | 2025, NeurIPS submission | 从线性化和局部曲率解释 LLM 微调 | 适合连接 NTK、LoRA 和 Hessian 分析 |

**使用建议：** Hessian 谱适合回答“模型现在处在什么几何状态”，不直接回答“哪个方向一定能提升下游任务”。top eigenvalue、trace、density、梯度噪声尺度应和验证损失、学习率、batch size 一起看。

## 6. Hessian/Fisher 辅助的 LLM 压缩、剪枝和量化

### 6.1 经典主线

| 论文                                                                                                                           | 年份/venue         | 方法与影响                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------- |
| [Deep Compression](https://doi.org/10.48550/arxiv.1510.00149)                                                                  | 2015               | 剪枝、量化、Huffman 编码的组合范式；不显式依赖 Hessian，但定义了后续压缩系统的基线    |
| [HAWQ: Hessian AWare Quantization](https://doi.org/10.1109/iccv.2019.00038)                                                    | 2019, ICCV         | 用 Hessian trace/谱敏感度做层级混合精度分配                                           |
| [HAWQ-V2](https://doi.org/10.48550/arxiv.1911.03852)                                                                           | 2019/2020          | 用 trace-weighted 指标改进层级 bit 分配                                               |
| [Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT](https://doi.org/10.1609/aaai.v34i05.6409)                     | 2020, AAAI         | 将 Hessian 敏感度用于 BERT 的超低比特量化                                             |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://doi.org/10.48550/arxiv.2210.17323) | 2022               | 将 Optimal Brain Quantization 的逆 Hessian 误差补偿做成可扩展的 LLM PTQ；工程影响极大 |
| [Optimal Brain Compression](https://doi.org/10.48550/arxiv.2208.11580)                                                         | 2022, NeurIPS 2022 | 统一二阶量化和剪枝，系统化 OBQ/OBS 的 LLM 版本                                        |
| [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://doi.org/10.48550/arxiv.2301.00774)           | 2023               | 用近似 Hessian 做一次性非结构化/半结构化剪枝，不需要再训练                            |
| [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://openreview.net/forum?id=xrk9g5vcXR)                | 2023, NeurIPS      | 将二阶误差、incoherence 和格码本结合，推动 2-bit LLM 量化                             |
| [QuIP#](https://arxiv.org/abs/2402.04396)                                                                                      | 2024               | 用 Hadamard incoherence 和更好的 lattice codebook 改进 QuIP                           |

### 6.2 需要一起比较的非 Hessian 基线

- **Wanda**：用权重幅值和输入激活的乘积做重要性，计算便宜；它不是完整二阶方法，但常作为 SparseGPT 的强基线。
- **SmoothQuant**：把激活 outlier 的量化难度迁移到权重；核心是 activation-aware scaling，不是 Hessian。
- **OmniQuant/LLM-QAT**：通过可学习量化参数或量化感知训练补偿误差。

这一区分很重要：如果论文写“sensitivity-aware”，需要确认 sensitivity 是 Hessian/Fisher、激活统计、梯度，还是简单幅值。

## 7. SVD 与低秩压缩：从矩阵分解到激活/任务感知

### 7.1 传统低秩压缩的桥梁

| 论文                                                                                                                  | 年份 | 贡献                                                      |
| --------------------------------------------------------------------------------------------------------------------- | ---: | --------------------------------------------------------- |
| [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736) | 2014 | 用低秩/线性结构分解网络层，奠定神经网络矩阵压缩范式       |
| [Language model compression with weighted low-rank factorization](https://doi.org/10.48550/arxiv.2207.00112)          | 2022 | 把激活/损失权重纳入语言模型低秩分解，而非只保留最大奇异值 |
| [TensorGPT](https://doi.org/10.48550/arxiv.2307.00526)                                                                | 2023 | 用 tensor-train 等结构压缩 LLM，拓展“矩阵 SVD”到张量分解  |

### 7.2 LLM 专门的 SVD/谱方法

| 论文                                                                                                                      | 年份/venue     | 核心思想                                                                     |
| ------------------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------- |
| [ASVD: Activation-aware Singular Value Decomposition](https://doi.org/10.48550/arxiv.2312.05821)                          | 2023           | 用激活统计重新缩放权重后做 SVD，避免普通谱能量和真实输出敏感度错位           |
| [SVD-LLM: Truncation-aware Singular Value Decomposition](https://doi.org/10.48550/arxiv.2403.07378)                       | 2024           | 把截断误差、层级 rank 和 LLM 输出重构结合                                    |
| [LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression](https://openreview.net/forum?id=mhI5nc5QwX) | 2024, ICML     | 认为 attention/MLP 子层不能用同一压缩策略，做结构化低秩分配                  |
| [SVD-LLM V2](https://doi.org/10.18653/v1/2025.naacl-long.217)                                                             | 2025, NAACL    | 改进 singular-value truncation 和层级分配                                    |
| [Dobi-SVD: Differentiable SVD for LLM Compression](https://openreview.net/forum?id=kws76i5XB8)                            | 2025, ICLR     | 让分解/截断进入可微优化，而不是纯 post-hoc 截断                              |
| [SVDQuant](https://openreview.net/forum?id=vWR3KuiQur)                                                                    | 2025, ICLR     | 用低秩分量吸收 outlier，再做低比特量化；主要面向 diffusion，但方法思想可迁移 |
| [ERC-SVD](http://arxiv.org/abs/2505.20112)                                                                                | 2025, preprint | 以误差约束控制 SVD 压缩                                                      |
| [SVD-Surgeon](http://arxiv.org/abs/2606.23568)                                                                            | 2026, preprint | 直接对奇异值做“手术式”重整，属于最新探索，影响力尚未稳定                     |
| [IO-SVD](http://arxiv.org/abs/2605.15626)                                                                                 | 2026, preprint | 对输入/输出做白化后再自适应 rank 压缩                                        |

### 7.3 Fisher/Hessian 加权 SVD

普通 SVD 只最小化 $\Vert W-\widetilde W\Vert_F$。若使用 Kronecker-Fisher 或输入协方差加权，则方向选择更接近任务损失。值得重点跟踪：

- [Generalized Fisher-Weighted SVD](http://arxiv.org/abs/2505.17974)：把可扩展 Kronecker-Fisher 近似用于 LLM 压缩。
- [Integrating Independent Layer-Wise Rank Selection with Low-Rank SVD Training](https://doi.org/10.24963/ijcai.2024/589)：层级独立 rank 选择和 SVD 训练。
- [QERA: an Analytical Framework for Quantization Error Reconstruction](https://openreview.net/forum?id=LB5cKhgOTu)：分析量化误差的低秩重构，连接 quantization 与 adapter。

这条线的核心问题不是“奇异值能否衰减”，而是：**保留哪一个谱方向，才能最大限度地保留校准分布上的 logits、loss 和下游任务能力？**

## 8. 低秩训练、LoRA 与 PEFT

### 8.1 从 LoRA 到梯度子空间

| 论文                                                                                                               | 年份/venue      | 低秩对象               | 关键点                                                                      |
| ------------------------------------------------------------------------------------------------------------------ | --------------- | ---------------------- | --------------------------------------------------------------------------- |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)                             | 2021/2022, ICLR | 权重更新 $\Delta W=BA$ | 冻结基座，仅训练低秩增量；PEFT 的事实基线                                   |
| [QLoRA](https://doi.org/10.48550/arxiv.2305.14314)                                                                 | 2023, NeurIPS   | 量化基座 + LoRA 更新   | 把 4-bit 基座训练带入实践                                                   |
| [AdaLoRA](https://arxiv.org/abs/2303.10512)                                                                        | 2023            | adapter 的 rank        | 根据重要性动态分配 rank，而非每层固定 rank                                  |
| [ReLoRA: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695)                            | 2023            | 训练更新               | 周期性合并/重置低秩更新，尝试逼近高秩训练                                   |
| [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://doi.org/10.48550/arxiv.2403.03507) | 2024            | 梯度                   | 投影优化器状态/梯度到低秩子空间，可做全参数训练，和 LoRA 的“低秩参数化”不同 |
| [Q-GaLore](http://arxiv.org/abs/2407.08296)                                                                        | 2024            | 量化梯度投影           | 将 GaLore 与 INT4 和 layer-adaptive rank 结合                               |
| [SwitchLoRA](https://doi.org/10.48550/arxiv.2406.06564)                                                            | 2024            | 动态低秩分量           | 通过切换低秩子空间增加有效表达能力                                          |

### 8.2 SVD 初始化和数据/任务感知子空间

| 论文                                                                                                                | 年份/venue           | 方法                                                                                |
| ------------------------------------------------------------------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------- |
| [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://doi.org/10.48550/arxiv.2404.02948)       | 2024                 | 用预训练权重的主奇异值/向量初始化 adapter；相比随机 LoRA 更快收敛、量化误差更易补偿 |
| [EVA: Parameter Efficient Fine-tuning via Explained Variance Adaptation](https://arxiv.org/abs/2410.07170)          | 2024/2025            | 用输入激活 PCA 的 explained variance 做 layer-wise rank allocation                  |
| [CorDA: Context-Oriented Decomposition Adaptation](https://openreview.net/forum?id=Gi00NVru6n)                      | 2024, NeurIPS        | 依据任务上下文/知识保留目标构造分解；是“数据分布决定子空间”的代表                   |
| [SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors](https://openreview.net/forum?id=uS0PwIBzC0)           | 2024, NeurIPS        | 直接训练一部分 singular vectors，而不是完整低秩因子                                 |
| [LoRA-XS](https://doi.org/10.3233/faia251185)                                                                       | 2025                 | 极少参数的低秩适配，探索 rank 和可训练参数的极限                                    |
| [DisLoRA](https://doi.org/10.18653/v1/2025.emnlp-main.694)                                                          | 2025, EMNLP Findings | 用 SVD 得到任务特定的正交基                                                         |
| [On the Optimization Landscape of Low Rank Adaptation Methods for LLMs](https://openreview.net/forum?id=pxclAomHat) | 2025, ICLR           | 直接分析 LoRA 类方法的非凸优化地形和秩限制                                          |

### 8.3 低秩与量化结合

| 论文                                                         | 年份/venue           | 作用                                                   |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------------------ |
| [LoftQ](https://doi.org/10.48550/arxiv.2310.08659)           | 2023/2024            | 用低秩分解初始化 LoRA 以补偿量化误差                   |
| [QA-LoRA](https://doi.org/10.48550/arxiv.2309.14717)         | 2023                 | 量化分组和 LoRA rank/结构联合设计                      |
| [LQ-LoRA](https://doi.org/10.48550/arxiv.2311.12023)         | 2023/2024, ICLR      | low-rank plus quantized matrix decomposition           |
| [DL-QAT](https://doi.org/10.18653/v1/2024.emnlp-industry.10) | 2024, EMNLP Industry | weight-decomposed low-rank quantization-aware training |
| [QERA](https://openreview.net/forum?id=LB5cKhgOTu)           | 2025, ICLR           | 用解析误差重构把量化残差压到低秩空间                   |

## 9. 训练分析和辅助训练到底该怎么选

| 目标                                   | 首选论文/工具                                       | 为什么                                                          |
| -------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| 看训练是否接近不稳定/edge of stability | Hessian top eigenvalue、PyHessian、谱密度、梯度噪声 | 比单看 loss 更能解释学习率和 batch size 的变化                  |
| 让预训练更省显存                       | GaLore；再比较 Sophia、AdaHessian、K-FAC            | GaLore 降低优化器状态；Sophia 使用廉价二阶估计                  |
| 做全参数二阶预条件                     | K-FAC、layer-wise natural gradient、AdaFisher       | 适合有足够算力和稳定统计量的训练，不适合盲目在线构造完整 Fisher |
| 一次性剪枝                             | SparseGPT、OBC；Wanda 作为便宜基线                  | Hessian 补偿参数相关性；Wanda 适合快速筛选                      |
| 低比特 PTQ                             | GPTQ；2-bit 看 QuIP/QuIP#；混合精度看 HAWQ/Q-BERT   | 分别对应误差补偿、极低 bit 和层级 bit 分配                      |
| 压缩权重而不是量化                     | ASVD、SVD-LLM、SVD-LLM V2、Dobi-SVD                 | 激活/截断感知通常比裸 SVD 更可靠                                |
| PEFT 微调                              | LoRA → AdaLoRA → PiSSA/EVA/CorDA                    | 先用基线，再根据数据分布选择初始化和 rank allocation            |
| 量化模型上做 PEFT                      | QLoRA、LoftQ、LQ-LoRA、QERA                         | 低秩更新用来重构量化残差                                        |
| 持续预训练/任务序列                    | EWC、Online Laplace/K-FAC、Fisher masking           | 用 Fisher 保护旧知识；需注意 Fisher 估计分布依赖                |

## 10. 关键争议和实验陷阱

1. **Hessian 不是 Fisher。** 负对数似然在模型分布下有关系，但 empirical Fisher 常常只是梯度外积的启发式近似。
2. **sharpness 不是参数化不变量。** 直接比较最大特征值或 trace 时，必须固定参数化、归一化和模型尺度。
3. **对角近似会丢掉补偿。** OBD/EWC 的 diagonal 便宜但忽略参数相关性；OBS/GPTQ/SparseGPT 的价值正是利用逆曲率进行误差补偿。
4. **SVD 能量不等于任务重要性。** 奇异值最大的方向可能对校准激活或下游 loss 不重要；需要 activation-aware、Fisher-weighted 或 task-aware 分解。
5. **校准分布决定结论。** GPTQ/ASVD/SVD-LLM 的 calibration data 与部署分布不匹配时，谱/敏感度排序会失效。
6. **低秩更新不是低秩梯度。** LoRA 限制 $\Delta W$ 的参数化；GaLore 投影梯度/优化器状态；PiSSA/EVA/CorDA 改的是初始子空间或 rank 分配，实验中不能混称“都在做 SVD”。
7. **层间异质性不可忽略。** attention projection、MLP、embedding、lm-head 的谱衰减和量化敏感度不同；统一 rank/bit 往往不是 Pareto 最优。
8. **二阶成本仍然真实存在。** LLM 的全参数 Fisher/Hessian 不可存储；实用方法依赖 HVP、块结构、随机 trace、低秩草图、激活统计或少量校准数据。

## 11. 推荐阅读路径

### 路径 A：想理解“为什么二阶量化/剪枝有效”

OBD → OBS → HAWQ → GPTQ → OBC → SparseGPT → QuIP/QuIP#。

### 路径 B：想理解“LLM 的 Hessian/Fisher 训练分析”

Sagun 2017 → Hessian Eigenvalue Density → PyHessian → Power-Law Hessian Spectrums → Sophia → K-FAC for Modern Architectures → Layer-Wise Natural Gradient/AdaFisher.

### 路径 C：想做 SVD/低秩压缩

Denton/Jaderberg 2014 → weighted low-rank language model compression → ASVD → SVD-LLM → SVD-LLM V2 → Dobi-SVD → Fisher-Weighted SVD.

### 路径 D：想做 PEFT 或低显存训练

LoRA → QLoRA → AdaLoRA → GaLore → PiSSA → EVA → CorDA → LoftQ/LQ-LoRA/QERA。

## 12. 我认为最值得优先复现的 10 篇

1. GPTQ：最清楚地展示了逆 Hessian 误差补偿在 LLM PTQ 中的工程价值。
2. SparseGPT：把同一套曲率思想推进到一次性剪枝。
3. Optimal Brain Compression：理解 pruning/quantization 的统一二阶框架。
4. HAWQ-V2：理解 Hessian trace 如何变成 mixed-precision bit allocation。
5. PyHessian：建立 Hessian 谱诊断工具链。
6. Sophia：观察二阶统计如何进入大规模预训练，而不承担完整 Newton 成本。
7. LoRA：低秩更新的基线和实验协议。
8. GaLore：区分“低秩参数化”和“低秩梯度/优化器状态”。
9. ASVD：理解激活统计如何改变 SVD 压缩排序。
10. PiSSA/EVA/CorDA（三选一或全做）：分别对应谱初始化、解释方差 rank 分配和上下文感知分解。

## 13. 进一步值得关注的开放问题

- 能否用统一的 Kronecker-Fisher/Hessian 度量同时决定 bit、结构稀疏度和 rank？
- 预训练阶段的 Hessian 谱能否可靠预测指令微调、RLHF/RLVR 后的敏感方向？
- 如何处理 Transformer 中 attention、MLP、normalization 和 KV cache 的不同曲率尺度？
- Fisher 子空间是否比 activation PCA 更稳定地支持跨任务、多语言和持续学习？
- 动态 rank/bit 分配能否在线更新而不让校准统计和训练吞吐成为瓶颈？
- 低秩压缩后的推理速度是否真正受益于硬件 kernel，而不仅是参数量减少？

## 14. 检索报告和原始结果

本次原始检索报告保存在 `paper-search-reports/`，其中最相关的几份是：

- `20260909-002950-394929-hessian-large-language-model-training-fisher-information-neural-.md`
- `20260909-003551-628728-sagun-hessian-eigenvalues-deep-learning-an-investigation-into-ne.md`
- `20260909-003624-795782-lora-low-rank-adaptation-adalora-galore-relora-dora-pissa-eva-ex.md`
- `20260909-003624-807664-hawq-hawq-v2-gptq-optimal-brain-quantization-sparsegpt-quip-q-be.md`
- `20260909-004548-987684-svd-llm-truncation-aware-singular-value-decomposition-asvd-activ.md`
- `20260909-004744-141473-fisher-pruning-neural-network-fisher-information-for-neural-netw.md`
