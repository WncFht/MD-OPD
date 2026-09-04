# 数据拉取记录（2026-09-04）

工作目录：`/Users/fanghaotian/Desktop/src/One-Shot-OPD`

## 环境准备

系统 Python 缺少 `pandas`，因此创建项目本地虚拟环境并安装数据处理依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install pandas pyarrow numpy
.venv/bin/python -m pip install datasets
```

## 实际执行的拉取与构建命令

```bash
# Multi-IF 源数据
bash data/prep/ms_fetch.sh datasets facebook/Multi-IF data/.cache/multiif
.venv/bin/python data/prep/make_multiif_train.py \
  --src data/.cache/multiif/multiIF_20241018.csv
.venv/bin/python data/prep/prepare_multiif_eval.py \
  --src data/.cache/multiif/multiIF_20241018.csv

# 数学训练、验证与 one-shot 数据
.venv/bin/python data/prep/prepare_dapo_math500.py --set all
.venv/bin/python data/prep/prepare_dapo_oneshot.py
.venv/bin/python data/prep/prepare_amc23_aime25.py all

# 代码评测数据
.venv/bin/python data/prep/prepare_code_eval.py --bench all

# xLAM 函数调用训练数据
bash data/prep/ms_fetch.sh datasets Salesforce/xlam-function-calling-60k data/.cache/xlam
.venv/bin/python data/prep/prepare_xlam_train.py \
  --src data/.cache/xlam/xlam_function_calling_60k.json

# TACO 代码训练数据（9 个 shard，断点续传）
TACO_CACHE=data/.cache/taco \
  .venv/bin/python data/prep/prepare_taco_train.py

# WildChat 原始 shard 与转换
bash data/prep/ms_fetch.sh datasets allenai/WildChat-1M data/.cache/wildchat data
WILDCHAT_DIR=data/.cache/wildchat/data \
  .venv/bin/python data/prep/prepare_wildchat_for_opd.py
```

## Prepare 脚本的处理内容

- `prepare_dapo_math500.py` 从 ModelScope 读取 DAPO-Math-17k 和 MATH-500；DAPO 按题目索引去重（上游每题重复 100 次），去掉统一的答案格式说明，保留 17,917 道题；MATH-500 将 JSONL 的题目和答案包装成标准 schema。
- `prepare_dapo_oneshot.py` 从 DAPO 训练 parquet 取第 22 行，并完整重复 64 次，生成 one-shot 训练文件。
- `prepare_amc23_aime25.py` 读取 AMC23 的 40 题，以及 AIME25 I/II 两个各 15 题的 JSONL，合并为 30 题，统一写成数学验证 schema。
- `prepare_code_eval.py` 将 HumanEval+ 和 MBPP+ 的函数题包装成带 Python 代码块提示的记录，并把测试代码/入口函数放入 `reward_model.ground_truth`；LCB v6 解码 public/private 测试用例（含 base64+zlib+pickle 格式），只保留有测试用例的题目，生成 175 题。
- `prepare_taco_train.py` 合并 9 个 TACO Arrow shard，过滤缺少题目或测试的记录，将测试用例和元数据整理为代码训练 schema，最终保留 24,701 题。
- `make_multiif_train.py` 读取 Multi-IF CSV 的第一轮 prompt、指令 ID 和参数，生成 4,501 条 IF 训练记录，并保留语言信息在 `extra_info.source` 中。
- `prepare_multiif_eval.py` 仅保留有第三轮 prompt 的记录，解析三轮 prompt、指令 ID 和参数，生成供 `eval/run_if_eval.py` 使用的 4,445 行评估文件。
- `prepare_xlam_train.py` 从 xLAM JSON 读取 query、工具列表和答案；过滤无 query、无工具或答案格式错误的记录，将工具 schema 格式化进 system prompt，保留 60,000 条函数调用训练记录。
- `prepare_wildchat_for_opd.py` 读取 14 个 WildChat shard，提取每个对话的首条用户消息，筛选第一轮、英文、非 toxic 内容，再保留长度 20–2000 字符的消息，写出 192,857 条通用对话记录。

上述脚本都将训练/验证记录写入统一的 `prompt`、`data_source`、`ability`、`reward_model` 和 `extra_info` 字段；`align_mopd_schema.py` 用于检查这些字段能否被 MOPD 拼接。

## 校验命令

```bash
.venv/bin/python data/prep/align_mopd_schema.py
```

schema 校验通过。主要输出行数如下：

| 文件                              |    行数 |
| --------------------------------- | ------: |
| `math/DAPO/train.parquet`         |  17,917 |
| `math/MATH-500/test.parquet`      |     500 |
| `math/DAPO-oneshot/t22.parquet`   |      64 |
| `math/AMC23/test.parquet`         |      40 |
| `math/AIME25/test.parquet`        |      30 |
| `code/TACO/train.parquet`         |  24,701 |
| `code/LCBv6/test.parquet`         |     175 |
| `code/HumanEvalPlus/test.parquet` |     164 |
| `code/MBPPPlus/test.parquet`      |     378 |
| `if/MultiIF/train.parquet`        |   4,501 |
| `if/MultiIF/eval_3turn.parquet`   |   4,445 |
| `agentic/xlam/train.parquet`      |  60,000 |
| `chat/WildChat/train.parquet`     | 192,857 |

WildChat 转换日志：14 个 shard 原始共 837,989 行；筛选首轮、英文、非 toxic 后为 316,996 行，再按长度（20–2000 字符）过滤，最终 192,857 行。

BFCL 未执行：`prepare_bfcl_eval.py` 要求可访问的 `gorilla-llm/Berkeley-Function-Calling-Leaderboard` 本地副本，当前环境未提供该目录。
