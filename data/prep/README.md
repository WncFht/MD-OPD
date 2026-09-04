# `data/prep/`

这些脚本从公开来源重新构建 `data/` 下的所有 parquet。仓库不附带数据集，因此这是获取数据的唯一受支持方式。

输出默认写入 `<repo>/data/<domain>/<dataset>/`，由脚本所在位置的 `_paths.py` 解析，也可以通过 `DATA_BASE` 覆盖。请在仓库根目录运行：

```bash
python data/prep/<script>.py [--out ...]
```

| 脚本                          | 输出                                                                    | 外部依赖                                                                                                                    |
| ----------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ms_fetch.sh`                 | 任意 ModelScope 模型或数据集仓库                                        | 无，使用普通 `curl`                                                                                                         |
| `prepare_dapo_math500.py`     | `math/DAPO/train.parquet`、`math/MATH-500/test.parquet`                 | 无，从 ModelScope 获取                                                                                                      |
| `prepare_amc23_aime25.py`     | `math/AMC23/test.parquet`（40 行）、`math/AIME25/test.parquet`（30 行） | 无，从 ModelScope 获取 `knoveleng/AMC-23` 和 `opencompass/AIME2025`                                                         |
| `prepare_taco_train.py`       | `code/TACO/train.parquet`                                               | 无，从 ModelScope 获取 `BAAI/TACO` 的 9 个 arrow shard（约 4.2 GB）到 `$TACO_CACHE`；`ms_fetch.sh` 也可用 curl 获取同样文件 |
| `prepare_code_eval.py`        | `code/{LCBv6,HumanEvalPlus,MBPPPlus}/test.parquet`                      | 无，从 ModelScope 获取 `test6.jsonl` 和 evalscope 的 `evalplus/{humanevalplus,mbppplus}` 镜像                               |
| `prepare_xlam_train.py`       | `agentic/xlam/train.parquet`                                            | `Salesforce/xlam-function-calling-60k` 的本地副本（ModelScope 提供，见下文）                                                |
| `prepare_bfcl_eval.py`        | `agentic/BFCL/<subset>/test.parquet`                                    | `gorilla-llm/BFCL` 的本地副本                                                                                               |
| `make_multiif_train.py`       | `if/MultiIF/train.parquet`（4501 行，仅第一轮）                         | `--src multiIF_20241018.csv`                                                                                                |
| `prepare_multiif_eval.py`     | `if/MultiIF/eval_3turn.parquet`（4445 行，全部轮次）                    | 同一 CSV                                                                                                                    |
| `prepare_wildchat_for_opd.py` | `chat/WildChat/train.parquet`                                           | `WILDCHAT_DIR` 指向 WildChat-1M shard（14 个 parquet，约 3.4 GB；ModelScope 提供，见下文）                                  |
| `prepare_dapo_oneshot.py`     | `math/DAPO-oneshot/t22.parquet`                                         | 无，从 `math/DAPO/train.parquet` 切片                                                                                       |
| `align_mopd_schema.py`        | schema 校验与重写                                                       | 无                                                                                                                          |

## 与参考文件校验

`prepare_dapo_math500.py --verify <parquet>` 会逐行比较，并分别报告字节级一致和语义等价的数量。

| 文件                         | 字节级一致                        | 等价        |
| ---------------------------- | --------------------------------- | ----------- |
| `math/DAPO/train.parquet`    | 17917/17917                       | 17917/17917 |
| `math/MATH-500/test.parquet` | 500/500                           | 500/500     |
| `code/LCBv6/test.parquet`    | 175/175（prompt 和 ground truth） | —           |

## 完整重建

`hf-mirror.com` 和 `huggingface.co` 在当前环境会发生 SNI 重置（curl 退出码 35/104，HTTP 000）。ModelScope 响应正常，并提供所有必需仓库。

```bash
# Code teacher（约 7.1 GB）
bash data/prep/ms_fetch.sh models agentica-org/DeepCoder-1.5B-Preview \
     "$MODEL_BASE/DeepCoder-1.5B-Preview"

# 数学训练集和验证集
python3 data/prep/prepare_dapo_math500.py --set all

# 代码验证集（test6.jsonl 约 134 MB，175 题；HumanEval+ 164；MBPP+ 378）
python3 data/prep/prepare_code_eval.py --bench all

# 代码训练集（9 个 arrow shard，约 4.2 GB；首次运行下载，中断后可续传）
export TACO_CACHE=<cache directory on persistent disk>
python3 data/prep/prepare_taco_train.py

# 函数调用训练集和验证集
bash data/prep/ms_fetch.sh datasets Salesforce/xlam-function-calling-60k data/.cache/xlam
python3 data/prep/prepare_xlam_train.py --src data/.cache/xlam/xlam_function_calling_60k.json
# BFCL 不在 ModelScope：--src 指向 huggingface.co 数据集副本
# gorilla-llm/Berkeley-Function-Calling-Leaderboard（BFCL_v3_*.json 及可能的 possible_answer/），
# 需要从可访问该网络的环境获取。
python3 data/prep/prepare_bfcl_eval.py --src <BFCL directory>

# WildChat 压力测试集（14 个 shard，约 3.4 GB；没有启动脚本读取）
bash data/prep/ms_fetch.sh datasets allenai/WildChat-1M data/.cache/wildchat data
WILDCHAT_DIR=data/.cache/wildchat/data python3 data/prep/prepare_wildchat_for_opd.py

# Multi-IF 源 CSV，同时用于训练集和评估集
bash data/prep/ms_fetch.sh datasets facebook/Multi-IF data/.cache/multiif
python3 data/prep/make_multiif_train.py
python3 data/prep/prepare_multiif_eval.py

# 验证 MOPD 拼接前提
python3 data/prep/align_mopd_schema.py
```

`ms_fetch.sh` 使用 `/usr/bin/curl` 通过普通 HTTP 传输，不会安装 `modelscope` 包，因为该包的依赖树会替换 `transformers 5.10.4`。不要使用 conda 自带的 curl：其 CA bundle 损坏，所有 HTTPS 请求都会以退出码 77 失败。下载使用 `curl -C -` 支持续传。

## 约束

**`data_source` 是验证器路由键，不是标签。** `prepare_code_eval.py` 必须写入 `livecodebench`。任何其他字符串都会落入数学验证器，Python 程序得分接近零，且不会报错。

**六个 MOPD 文件必须共享一个 schema**，否则 `datasets` 拼接时会报 “The features can't be aligned”。所有生成脚本都通过 `align_mopd_schema.write_aligned()` 写出。运行 `python data/prep/align_mopd_schema.py` 验证。

**struct 字段顺序不是差异。** 生成 split 时 `datasets` 会标准化，因此 TACO 的 `(role, content)` 与 DAPO 的 `(content, role)` 可以正确拼接。验证器会报告顺序差异，但不会因此失败。

**Multi-IF 训练集和评估集按设计存在重叠。** 两者来自同一 CSV，训练集是评估对话的第一轮。这符合参考设置，报告 Multi-IF 分数时应同时说明这一点。
