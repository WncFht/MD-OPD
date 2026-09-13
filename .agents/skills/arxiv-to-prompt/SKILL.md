---
name: arxiv-to-prompt
description: 用 `uvx arxiv-to-prompt` 拉取并展平 arXiv 论文的 LaTeX 源码。在阅读、解释、引用或基于论文 TeX 提问时使用（arXiv ID 或 abs URL）。不要用于论文检索。
---

# arxiv-to-prompt

始终用 `uvx` 调用，不要 `pip install`。默认把展平后的 LaTeX 打到 stdout。会自动取最新版本并缓存。

```bash
uvx arxiv-to-prompt <arxiv_id或URL> [选项]
```

## 输入

`arxiv_id` 与 `--local-folder` 二选一。

| 参数                  | 含义                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| `arxiv_id`            | 位置参数。arXiv ID（如 `2303.08774`）或 abs URL（如 `https://arxiv.org/abs/2303.08774`）。不必写版本号。 |
| `--local-folder PATH` | 处理本地 TeX 目录，不再下载。此时不要再给 `arxiv_id`。                                                   |

## 裁剪输出

未指定时输出全文（含注释与附录）。读论文时默认加 `--no-comments --no-appendix`，除非用户要完整源码。

| 参数              | 含义                                                                                            |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `--no-comments`   | 去掉 LaTeX 注释                                                                                 |
| `--no-appendix`   | 去掉 appendix 及其之后的全部内容                                                                |
| `--abstract`      | 只输出摘要文本                                                                                  |
| `--expand-macros` | 把 `\newcommand` 等宏内联展开                                                                   |
| `--section NAME`  | 只抽指定章节，可重复。重名时用路径：`--section "Fine-tuning > RLHF Results > Human Evaluation"` |
| `--list-sections` | 列出全部章节（子节缩进），不输出正文                                                            |

```bash
uvx arxiv-to-prompt 2303.08774 --no-comments --no-appendix
uvx arxiv-to-prompt 2307.09288 --list-sections
uvx arxiv-to-prompt 2307.09288 --section "Introduction" --section "Pretraining"
uvx arxiv-to-prompt 2303.08774 --abstract
uvx arxiv-to-prompt 2303.08774 --expand-macros --no-comments
```

章节名歧义时，工具会提示候选路径，按提示改用 `父节 > 子节` 即可。

## 其他输出

| 参数             | 含义                                                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--figure-paths` | 输出解析后的图片路径，而不是 LaTeX 正文。可与 `--no-appendix --no-comments` 合用，排除附录和被注释掉的图                          |
| `--token-count`  | 只打印 tiktoken token 数，不打印正文。需要 extra：`uvx --from 'arxiv-to-prompt[tokens]' arxiv-to-prompt 2303.08774 --token-count` |
| `-c` / `--copy`  | 写入剪贴板而非 stdout。Agent 不要用，必须读 stdout                                                                                |

```bash
uvx arxiv-to-prompt 2303.08774 --figure-paths --no-appendix --no-comments
uvx --from 'arxiv-to-prompt[tokens]' arxiv-to-prompt 2303.08774 --no-comments --no-appendix --token-count
```

## 缓存与下载

默认缓存目录：`~/.cache/arxiv-to-prompt`。同一篇论文再次调用会走缓存。

| 参数               | 含义                                         |
| ------------------ | -------------------------------------------- |
| `--cache-dir DIR`  | 自定义缓存目录                               |
| `--force-download` | 忽略缓存，强制重新下载                       |
| `--lock-timeout N` | 等另一进程下完同一篇的锁超时（秒）。默认 120 |

```bash
uvx arxiv-to-prompt 2303.08774 --force-download
uvx arxiv-to-prompt --local-folder /path/to/tex
uvx arxiv-to-prompt 2303.08774 --cache-dir /tmp/arxiv-cache --lock-timeout 300
```

## 杂项

| 参数            | 含义                                |
| --------------- | ----------------------------------- |
| `-h` / `--help` | 官方帮助。本 skill 已覆盖，不必再跑 |
| `--version`     | 打印工具版本                        |
