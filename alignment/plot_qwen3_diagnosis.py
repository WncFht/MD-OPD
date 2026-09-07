#!/usr/bin/env python3
"""Qwen3 师生对 OPD 平坦 gap 诊断图（docs/2026-09-07-qwen3-pair-opd-flat-gap.md）。

四张图，输出到 alignment/figures/qwen3/：
  fig1_eval_scores.png        三档 Qwen3 在 AIME24/25、AMC23 的 mean@16（justrl eval 口径）
  fig2_per_problem_scatter.png 逐题 pass@16：学生 vs 教师，可教区间可视化
  fig3_opd_training_curves.png OPD 训练关键指标：token gap 与 top-k overlap，Qwen3 对 vs DeepSeek 对
  fig4_val_curves.png          训练中 val mean@16：Qwen3 学生起点即平台 vs DeepSeek 学生持续上升

数据源：
  - justrl eval：/Users/fanghaotian/Desktop/src/OPD/justrl_eval_outputs/thinking（逐题 jsonl，
    用 OPD/scripts/val/eval/utils.grade_answer_verl 判分，规则分与官方 grading_results.json 一致）
  - wandb offline run：um5vzc6m（Qwen3 对）、mdxt04lm（DeepSeek 对基线），本地 CSV 在 alignment/runs/
  - val 成绩：wandb val-core/*/acc/mean@16，拉取后硬编码在下方 VAL 表
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "alignment" / "runs"
DEFAULT_FIG_DIR = REPO_ROOT / "alignment" / "figures" / "qwen3"

sys.path.insert(0, str(REPO_ROOT / ".agents" / "skills" / "plot-skill" / "scripts"))

from style_presets import (  # noqa: E402
    NEUTRALS,
    apply_frame,
    apply_grid,
    apply_rc,
    series_colors,
    series_markers,
)

THEME = "tableau-paper"

# 中文字体（macOS）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

TASKS = ["aime24", "aime25", "amc23"]
TASK_LABELS = {"aime24": "AIME24", "aime25": "AIME25", "amc23": "AMC23"}
MODELS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
MODEL_LABELS = {
    "Qwen3-1.7B": "Qwen3-1.7B（学生）",
    "Qwen3-4B": "Qwen3-4B（教师）",
    "Qwen3-8B": "Qwen3-8B",
}


def per_problem_pass(model: str, task: str, eval_dir: Path, opd_eval_dir: Path) -> dict[str, int]:
    """逐题 pass@16（example_id 与 prompt 已验证一一对应）。"""
    sys.path.insert(0, str(opd_eval_dir))
    from utils import grade_answer_verl

    path = eval_dir / model / f"{task}_t0.7_p0.95_n16-MNT31744.jsonl"
    per: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    for line in open(path):
        d = json.loads(line)
        tot[d["example_id"]] += 1
        if grade_answer_verl(d["response"], str(d["answer"])):
            per[d["example_id"]] += 1
    return {k: per.get(k, 0) for k in tot}


# ---------------------------------------------------------------- fig1: eval 成绩
def fig1(passes: dict[str, dict[str, dict[str, int]]]) -> None:
    means = {m: [np.mean(list(passes[m][t].values())) / 16 for t in TASKS] for m in MODELS}
    apply_rc(THEME)
    colors = series_colors(THEME, len(MODELS))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(TASKS))
    w = 0.24
    for i, m in enumerate(MODELS):
        bars = ax.bar(
            x + (i - 1) * w,
            means[m],
            w * 0.92,
            color=colors[i],
            label=MODEL_LABELS[m],
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, means[m]):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=NEUTRALS["ink"],
            )
    ax.set_xticks(x, [TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("mean@16 准确率")
    ax.set_ylim(0, 1.0)
    ax.set_title("Qwen3 三档模型解题成绩（justrl eval：T=0.7, top_p=0.95, n=16, thinking）")
    apply_frame(ax, "closed")
    apply_grid(ax, "y-only")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_eval_scores.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig2: 逐题 scatter
def fig2(passes: dict[str, dict[str, dict[str, int]]]) -> None:
    apply_rc(THEME)
    colors = series_colors(THEME, 3)
    marks = series_markers(THEME, 3)
    teachable = {"aime24": "21/30", "aime25": "19/30", "amc23": "26/83"}
    fig, ax = plt.subplots(figsize=(6.0, 6.6))
    for i, t in enumerate(TASKS):
        sx = np.array(list(passes["Qwen3-1.7B"][t].values()), dtype=float) / 16
        ty = np.array(list(passes["Qwen3-4B"][t].values()), dtype=float) / 16
        ax.scatter(
            sx,
            ty,
            s=26,
            alpha=0.65,
            color=colors[i],
            marker=marks[i],
            label=f"{TASK_LABELS[t]}（可教 {teachable[t]}）",
            edgecolors="white",
            linewidths=0.4,
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=NEUTRALS["baseline"],
        lw=1.0,
        ls="--",
        label="y = x（师生持平）",
    )
    ax.fill_between([0, 1], [0, 1], [1, 1], color=colors[1], alpha=0.06)
    ax.text(
        0.36,
        0.80,
        "阴影区：教师会、学生不会\n—— OPD 信号的来源",
        fontsize=10,
        color=NEUTRALS["ink"],
        ha="center",
    )
    ax.annotate(
        "双双全对：OPD 无信号",
        xy=(0.985, 0.975),
        xytext=(0.56, 0.56),
        fontsize=9.5,
        color=NEUTRALS["ink"],
        arrowprops=dict(arrowstyle="->", color=NEUTRALS["baseline"], lw=1.0),
    )
    ax.set_xlabel("学生 Qwen3-1.7B 逐题 pass@16")
    ax.set_ylabel("教师 Qwen3-4B 逐题 pass@16")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("逐题难度分布：可教区间集中在 AIME 段", fontsize=11)
    apply_frame(ax, "closed")
    apply_grid(ax, "full-dotted")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_per_problem_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig3: OPD 训练曲线
RUN_PAIRS = {
    "Qwen3 对（1.7B→4B，异源）": "um5vzc6m",
    "DeepSeek 对（1.5B→1.5B，同源）": "mdxt04lm",
}


def fig3() -> None:
    apply_rc(THEME)
    colors = series_colors(THEME, 2)
    marks = series_markers(THEME, 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    metrics = [
        (
            "opd/token_reward_mean",
            "opd/token_reward_mean（师生逐 token gap）",
            "平均每 token 师生 logp gap",
        ),
        (
            "opd/top_k_overlap_ratio",
            "opd/top_k_overlap_ratio（师生 top-k 重叠率）",
            "top-k 重叠率",
        ),
    ]
    for ax, (key, title, ylab) in zip(axes, metrics):
        for i, (label, rid) in enumerate(RUN_PAIRS.items()):
            df = pd.read_csv(RUNS_DIR / f"{rid}__history.csv").sort_values("_step")
            s = df[["_step", key]].dropna()
            every = max(1, len(s) // 60)
            s2 = s.iloc[::every]
            ax.plot(
                s2["_step"],
                s2[key].abs() if "reward" in key else s2[key],
                color=colors[i],
                marker=marks[i],
                ms=3.2,
                lw=1.6,
                markevery=3,
                label=label,
            )
        ax.set_xlabel("训练步")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        apply_frame(ax, "closed")
        apply_grid(ax, "y-only")
        ax.legend(loc="best", fontsize=8.5)
    axes[0].set_title("师生逐 token logp gap（|token_reward_mean|）", fontsize=10.5)
    fig.suptitle(
        "Qwen3 对：gap 大且不收敛，重叠率不升；DeepSeek 对：gap 收敛、重叠率上升",
        fontsize=11.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_opd_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig4: val 曲线
# wandb val-core/*/acc/mean@16（2026-09-07 拉取；um5vzc6m 到 step 140 仍在跑）
VAL = {
    "um5zc6m": {
        "MATH-500": (
            [20, 40, 60, 80, 100, 120, 140],
            [0.897, 0.889, 0.888, 0.888, 0.888, 0.889, 0.888],
        ),
        "AMC23": (
            [20, 40, 60, 80, 100, 120, 140],
            [0.852, 0.819, 0.830, 0.814, 0.836, 0.830, 0.841],
        ),
        "AIME25": (
            [20, 40, 60, 80, 100, 120, 140],
            [0.385, 0.331, 0.346, 0.377, 0.340, 0.333, 0.362],
        ),
    },
    "mdxt04lm": {
        "MATH-500": (
            [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 279],
            [
                0.842,
                0.862,
                0.867,
                0.854,
                0.836,
                0.844,
                0.859,
                0.862,
                0.866,
                0.873,
                0.871,
                0.871,
                0.874,
                0.877,
            ],
        ),
        "AMC23": (
            [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 279],
            [
                0.752,
                0.802,
                0.809,
                0.830,
                0.794,
                0.800,
                0.855,
                0.859,
                0.858,
                0.853,
                0.873,
                0.856,
                0.855,
                0.853,
            ],
        ),
        "AIME25": (
            [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 279],
            [
                0.252,
                0.283,
                0.285,
                0.317,
                0.292,
                0.319,
                0.310,
                0.321,
                0.331,
                0.331,
                0.342,
                0.352,
                0.333,
                0.331,
            ],
        ),
    },
}
RUN_LABELS = {
    "um5zc6m": "Qwen3-1.7B 学生（um5vzc6m）",
    "mdxt04lm": "DeepSeek-R1-Distill-1.5B 学生（mdxt04lm）",
}


def fig4() -> None:
    apply_rc(THEME)
    colors = series_colors(THEME, 3)
    marks = series_markers(THEME, 3)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    for ax, (rid, title) in zip(
        axes,
        [
            ("um5zc6m", "Qwen3 对：step 20 起全程平台"),
            ("mdxt04lm", "DeepSeek 对：持续上升"),
        ],
    ):
        for i, task in enumerate(["MATH-500", "AMC23", "AIME25"]):
            steps, vals = VAL[rid][task]
            ax.plot(steps, vals, color=colors[i], marker=marks[i], ms=4, lw=1.6, label=task)
        ax.set_xlabel("训练步")
        ax.set_title(title, fontsize=10.5)
        apply_frame(ax, "closed")
        apply_grid(ax, "y-only")
        ax.set_ylim(0.2, 1.0)
    axes[0].set_ylabel("val mean@16（valid 口径：T=0.7, top_p=0.9）")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle(
        "训练中验证成绩：Qwen3 学生起点即高位平台，DeepSeek 学生被训练持续抬升",
        fontsize=11.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_val_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--opd-eval-dir", type=Path, required=True)
    args = parser.parse_args()
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    global FIG_DIR, RUNS_DIR
    FIG_DIR, RUNS_DIR = fig_dir, args.runs_dir
    cache = fig_dir / "per_problem_pass_cache.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
    else:
        raw = {}
        for m in MODELS:
            raw[m] = {t: per_problem_pass(m, t, args.eval_dir, args.opd_eval_dir) for t in TASKS}
        cache.write_text(json.dumps(raw, indent=1))
        print(f"[cache] per-problem pass -> {cache}")
    passes = raw
    fig1(passes)
    fig2(passes)
    fig3()
    fig4()
    print(f"[done] 4 figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
