# 运行环境约定

## 机器拓扑

- 本地是 Mac，只用于写代码和查看结果，不运行 GPU 实验。
- 远程开发机 GPU 较少；开发机上的项目目录是 `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/OPD-Shot-OPD`。
- GPU 任务通过开发机执行 `hope run *.hope` 提交到集群，通常使用 4 张 GPU。
- 开发机与集群共享上述 DolphinFS 目录；队列文件和结果无需额外同步进程，集群可直接读取。
- Agent 无法直接访问开发机或集群；不要假设远端文件已同步到 Mac，收尾时说明需要同步的文件。

## 实验文档

- 将 `docs/` 视为可读性实验追踪和项目笔记的主目录。
- `docs/` 下顶层带日期的笔记必须使用 `YYYY-MM-DD-<slug>.md` 格式；不要使用 `article.md`、`周报.md` 或 `*.zh.md` 这类别名。

## 脚本

- 少使用 `export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}` 这种覆盖写法，多使用直接写默认值的写法 `export MINI_BATCH_SIZE=64`
- 不要写 `set -euo pipefail`，可以写 `set -eo pipefail`

文献引用一律用脚注：正文在作者-年份或论文名后紧跟 `[^key]`（键为小写 ASCII 与连字符，如 `[^zhang20]`；同一文献可多处引用），行内不放链接；文末设 `### 参考文献` 节，逐行写 `[^key]: 作者. 标题. venue 年份. [arXiv:编号](链接)`，编号由渲染器按首次引用自动生成，venue 只写核实过的。
