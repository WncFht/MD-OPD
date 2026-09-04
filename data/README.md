# `data/`

所有启动脚本读取的 parquet 数据。`_common.sh` 默认将 `DATA_BASE` 设置为
`${OPD_ROOT}/data`，因此新克隆的仓库无需额外配置数据路径。

仓库不附带任何数据集。除本文件和 `smoke/` 下的两个 fixture 外，`data/` 均被
git 忽略。下表中除 `smoke/` 外的所有数据，都必须使用 [`prep/`](prep/README.md)
中的脚本从公开来源重新构建。表中的行数、`data_source` 和 `ability` 是正确构建后的预期值。

## 目录结构

```
data/<domain>/<dataset>/{train,test}.parquet
```

| 目录       | 数据集                                                                       |                行数 | `data_source`   | `ability`               | 用途                                                  |
| ---------- | ---------------------------------------------------------------------------- | ------------------: | --------------- | ----------------------- | ----------------------------------------------------- |
| `math/`    | `DAPO/train.parquet`                                                         |               17917 | `DAPO`          | `math`                  | 数学训练集                                            |
|            | `MATH-500/test.parquet`                                                      |                 500 | `MATH-500`      | `math`                  | 数学验证集                                            |
|            | `DAPO-oneshot/t22.parquet`                                                   |                  64 | `DAPO`          | `math`                  | 1-shot 训练集：DAPO 第 22 题重复 64 次                |
| `code/`    | `TACO/train.parquet`                                                         |               24701 | `taco`          | `code`                  | 代码训练集                                            |
|            | `LCBv6/test.parquet`                                                         |                 175 | `livecodebench` | `code`                  | 代码验证集                                            |
| `if/`      | `MultiIF/train.parquet`                                                      |                4501 | `MultiIF`       | `instruction_following` | IF 训练集，仅第一轮                                   |
|            | `MultiIF/eval_3turn.parquet`                                                 |                4445 | —               | —                       | IF 验证集，由 `eval/run_if_eval.py` 读取，verl 不读取 |
| `agentic/` | `xlam/train.parquet`                                                         |               60000 | `xlam`          | `function_calling`      | 函数调用训练集                                        |
|            | `BFCL/{simple,multiple,parallel,parallel_multiple,live_simple}/test.parquet` | 400/200/200/200/258 | `bfcl_*`        | `function_calling`      | 函数调用验证集，五个子集                              |
| `chat/`    | `WildChat/train.parquet`                                                     |              192824 | `WildChat`      | `general`               | 没有启动脚本读取                                      |
| `smoke/`   | 两个 fixture                                                                 |            16 / 192 | mixed           | mixed                   | `recipe/smoke.sh` 已移除，当前未使用                  |

不存在 `template/` 目录。`MODE=template` 不读取语料：`TemplatePromptDataset` 会生成占位行，并忽略 `data_files`。

## `data_source` 决定验证器分支

`opd_mixed_verify` 按 `data_source` 的精确成员关系路由，数学分支是兜底分支。字符串写错不会报错，而是落入数学验证器，导致 Python 程序得分接近零。

## MOPD 的 schema 要求

MOPD 将数学、代码和 IF 训练集作为一个 `data.train_files`，将三个验证集作为一个 `data.val_files` 传入。`datasets` 要求拼接的数据逐字段拥有完全一致的 schema。

- 六个文件通过 `align_mopd_schema.write_aligned()` 写出，schema 定义在 `align_mopd_schema.py` 中。
- 不带参数运行 `python data/prep/align_mopd_schema.py` 可进行校验。
- struct 字段顺序不构成差异；`datasets` 生成 split 时会进行标准化。
- `agentic/` 和 `chat/` 被排除在外：函数调用单独训练和验证，且没有启动脚本读取 WildChat。
- `if/MultiIF/eval_3turn.parquet` 被排除，且文件名不是 `test.parquet`。它每行包含三个 prompt，没有 `reward_model` 列，仅由 `eval/run_if_eval.py` 读取。
