# ✨快速开始

## 环境配置

本项目通过扩展 [veRL](https://github.com/volcengine/verl) 实现 OPD 和 MOPD。启动脚本使用仓库内 [`verl/`](verl/) 目录中的 vendored 版本，因此新 clone 后不需要另外安装 veRL，只需安装其依赖。

```text
Python 3.12      vllm 0.28.0 (torch 2.13.0  transformers 5.10.4)      verl 0.9.0
```

```bash
git clone https://github.com/Thinking-Space/One-Shot-OPD.git
cd One-Shot-OPD

export MODEL_BASE=/your/models          # student 和 teacher checkpoint 所在目录
export CONDA_ENV_BIN=/your/env/bin      # vllm >= 0.18，transformers 5.x
```

- 文中的参考结果来自单台 8-GPU 节点（H100/A100 80 GB）；teacher worker 与 actor 共用 GPU 资源池。
- 仓库不附带数据集。数据由 [`data/prep/`](data/prep/README.md) 根据公开来源重新构建；预期目录结构、行数以及 `data_source` 取值见 [`data/README.md`](data/README.md)。

## 训练

| 脚本 | Reward | Teacher 数量 |
| --- | --- | --- |
| `grpo.sh` | 基于规则的 verifier，按每个 prompt 的 `n` 个样本分组 | 无 |
| `opd.sh` | teacher reverse KL | 1 个 |
| `mopd.sh` | teacher reverse KL，按每行的 `ability` 列路由 | 多个 |

```bash
bash recipe/grpo.sh                 # baseline
bash recipe/opd.sh                  # OPD，完整数据
MODE=oneshot bash recipe/opd.sh     # One-shot OPD
MODE=template bash recipe/opd.sh    # 由 student 自己写输入
bash recipe/mopd.sh                 # MOPD，三个 teacher
```

`DOMAIN` 选择 teacher 和 parquet，`MODE` 选择读取哪种 prompt；两者相互独立。命令行末尾追加的内容会原样传给 Hydra。

```bash
DOMAIN=code bash recipe/opd.sh              # math（默认）/ code / fc / if
DOMAIN=fc MODE=oneshot bash recipe/opd.sh   # 只有 math 提供 1-shot corpus
MODE=oneshot LOG_PROB_TOP_K=16 bash recipe/opd.sh trainer.total_training_steps=10
```

对于 MOPD，每个 teacher 都以 `名称:路径` 的形式传给 `multi_teacher_args`。增加 teacher 时，需要增加路径变量，并同时扩展路由映射和调用：

```bash
export TEACHER_AGENTIC_PATH=${TEACHER_AGENTIC_PATH:-${MODEL_BASE}/MyAgentTeacher}
export ABILITY_TO_TEACHER='{math:math,code:code,instruction_following:if,function_calling:agentic}'

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(multi_teacher_args \
        "math:${TEACHER_MATH_PATH}" \
        "code:${TEACHER_CODE_PATH}" \
        "if:${TEACHER_IF_PATH}" \
        "agentic:${TEACHER_AGENTIC_PATH}") \
    ...
```

<summary><b>关键参数</b></summary>

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DOMAIN` | `math` | `math` / `code` / `fc` / `if`，选择 teacher 和数据集 |
| `MODE` | `full` | `full` / `oneshot` / `template`，选择数据变体 |
| `ACTOR_MODEL_PATH` | 按 `DOMAIN` 决定 | 要训练的 student（policy）模型 |
| `TEACHER_MODEL_PATH` | 按 `DOMAIN` 决定 | 提供 token-level reward 的冻结 teacher |
| `MODEL_BASE` | 必填 | 两个模型路径所基于的目录 |
| `N_RESPONSES` | `1` | 每个 prompt 生成的 rollout 数量 |
| `LOG_PROB_TOP_K` | `0` | 计算 token reward 时保留的 top-K token；为 `0` 时退回到 sampled-token OPD |
| `METRIC_TOP_K` | `0` | 同样的计算仅用于**观测**；reward 仍是普通 reverse KL |
| `TOP_K_STRATEGY` | `only_stu` | 支持集：`only_stu` / `only_tch` / `intersection` / `union` / `union-intersection` |
| `REWARD_WEIGHT_MODE` | `student_p` | token 加权：`student_p` / `teacher_p` / `none` |
| `VAL_N` / `VAL_TEMPERATURE` / `VAL_TOP_P` | `16` / `0.7` / `0.9` | 验证采样设置，产生 `avg@16` |

> 两个 top-K 参数含义不同：`LOG_PROB_TOP_K` 会改变训练 reward；`METRIC_TOP_K` 只用于测量。

<summary><b>论文中 OPD 使用的默认超参数</b></summary>

| 项目 | 值 |
| --- | --- |
| Rollout batch size | 64 |
| Mini batch size | 64 |
| 每个 prompt 的 response 数 | 1 |
| KL 系数 | 0.0 |
| LogProb top-k | 0（默认）/ 16 |
| Top-k 策略 | student top-k |
| Loss 聚合 | `token-mean` |
| Training temperature | 1.0 |
| Top-p | 1.0 |
| Optimizer | AdamW（beta1 0.9，beta2 0.999，weight decay 0.01） |
| Learning rate | 1e-6 |
| Gradient clip norm | 1.0 |
| 最大 prompt 长度 | 1024（math）；4096（code、IF、agentic） |
| 最大 response 长度 | 7680（math、code、IF）；2048（agentic） |

## 验证

```bash
VAL_ONLY=True bash recipe/opd.sh    # 用当前权重打分，然后退出
```

### 训练期间能拿到哪些 validation metrics？

可以，不需要等训练结束。`TEST_FREQ` 默认是 `20`，训练循环每 20 个 global steps（以及最后一步）执行一次验证；`VAL_BEFORE_TRAIN=True` 时还会在训练开始前先验证一次。验证过程使用当前 student 权重，以 `VAL_N=16`、temperature `0.7`、top-p `0.9` 进行采样，然后调用 [`opd_mixed_verify.py`](verl/verl/utils/reward_score/opd_mixed_verify.py) 中按领域选择的 verifier。

日志（console/SwanLab）中的核心指标通常形如：

- `val-core/<data_source>/acc/mean@16`：每个样本的准确率/通过率的 16 次采样平均值，即 README 里所说的 `avg@16`。数学、代码和函数调用的 verifier 都会将结果写入相应的 `acc` 或 `reward` 变量。
- `val-aux/<data_source>/<变量>/mean@16`、`std@16`：辅助变量的均值和标准差。
- `val-aux/.../best@N/mean`、`worst@N/mean`：对每个 prompt 的 N 次采样取最好/最差结果后，再跨 prompt 平均；若 verifier 提供 `pred`，还会有 `maj@N/mean`（多数投票）。N 会从 2、4、8 增长到 16。
- 多轮数据还会有 `val-aux/num_turns/{min,max,mean}`；验证生成结果也可通过配置写入 JSONL。

每次验证产生的是当前 checkpoint/当前内存权重的即时结果，并不会参与梯度更新。`VAL_ONLY=True` 只是把同一套验证逻辑放到训练前执行一次并退出，适合给已有 checkpoint 单独打分。

论文中的完整评测协议如下；它们不完全等同于训练期间的验证：

| 领域 | Benchmark | 协议 | Response 上限 |
| --- | --- | --- | --- |
| 数学 | MATH-500、AMC 2023、AIME 2025 | `avg@16` | 31,744 |
| 代码 | LiveCodeBench v6 | `avg@3`，官方基于执行的 evaluator | 65,536 |
| 指令跟随 | Multi-IF | 八种语言的最终轮得分平均 | 每轮 16,384 |
| Agentic 工具使用 | BFCL v3 | 评测子集上的 `avg@8` | 4,096 |

#### Multi-IF

Multi-IF 是多轮任务：第 2 轮 prompt 依赖第 1 轮回答，而训练内置验证循环只能生成一次，因此无法完成该协议。它在 verl 外部运行；IF 训练会将 `VAL_FILES='[]'`，所以训练期间没有 IF validation 数值，需使用 `eval/run_if_eval.py`：

```bash
# 8 GPUs，每个 GPU 一个 vLLM engine（1.5B 模型，无 tensor parallelism）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 eval/run_if_eval.py multiif --model <hf-dir> --out results/multiif_s50

# 已有生成结果重新打分（修改 scorer 或聚合方式时无需 GPU）
python3 eval/run_if_eval.py rescore --out results/multiif_s50

# Smoke test，几分钟即可完成
python3 eval/run_if_eval.py multiif --model <hf-dir> --out results/smoke --limit 24
```
