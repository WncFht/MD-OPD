"""Plot M4 top-k reward distance and geometric absorption diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.metrics import effective_step, geometric_absorption
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment" / "runs"
TOPK_COLUMN = "opd/top_k_reward_mean"


def topk_metrics(frame):
    """Return step, positive top-k distance, normalized distance, and absorption."""
    if TOPK_COLUMN not in frame:
        raise KeyError(f"history has no {TOPK_COLUMN!r} column")
    steps = frame["_step"].to_numpy(dtype=float)
    distance = -frame[TOPK_COLUMN].to_numpy(dtype=float)
    baseline_rows = steps == BASELINE_STEP
    if not baseline_rows.any():
        raise ValueError("history has no step-30 baseline")
    baseline = float(distance[baseline_rows][0])
    return steps, distance, distance / baseline, geometric_absorption(steps, distance)


def plot_topk_diagnostics() -> Path:
    """Render top-k distance and absorption on step/folded axes."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for run_id, spec in M4_RUNS.items():
        frame = load_run_history(run_id, RUNS).frame
        steps, distance, relative, absorption = topk_metrics(frame)
        folded = effective_step(steps, spec["scale"])
        folded_baseline = BASELINE_STEP * spec["scale"]
        valid = np.isfinite(absorption)
        plot_series(axes[0, 0], steps, distance, color=spec["color"], linewidth=1.5, label=spec["label"])
        plot_series(
            axes[0, 1],
            folded,
            distance,
            color=spec["color"],
            linewidth=1.5,
            label=spec["label"],
            baseline=folded_baseline,
        )
        plot_series(axes[1, 0], steps, relative, color=spec["color"], linewidth=1.5)
        plot_series(axes[1, 1], folded, relative, color=spec["color"], linewidth=1.5, baseline=folded_baseline)
        plot_series(axes[2, 0], steps[valid], absorption[valid] * 100, color=spec["color"], linewidth=1.5)
        plot_series(
            axes[2, 1],
            folded[valid],
            absorption[valid] * 100,
            color=spec["color"],
            linewidth=1.5,
            baseline=folded_baseline,
        )
        for row in range(3):
            axes[row, 1].axvline(folded_baseline, color=spec["color"], ls=":", lw=0.7, alpha=0.5)

    axes[0, 0].set_title(r"top-$k$ distance on training step")
    axes[0, 1].set_title(r"top-$k$ distance on folded time")
    axes[1, 0].set_title(r"top-$k$ normalized distance on training step")
    axes[1, 1].set_title(r"top-$k$ normalized distance on folded time")
    axes[2, 0].set_title(r"top-$k$ absorption rate on training step")
    axes[2, 1].set_title(r"top-$k$ absorption rate on folded time")
    axes[0, 0].set_ylabel(r"$d_t^{\mathrm{top\text{-}k}}$ (nats/token)")
    axes[1, 0].set_ylabel(r"$d_t/d_{30}$")
    axes[2, 0].set_ylabel(r"$v_t$ (% per step)")
    axes[2, 0].set_xlabel("training step $t$")
    axes[2, 1].set_xlabel(r"effective time $u=(lr/10^{-6})t$")
    for row in axes:
        mark_baseline(row[0])
        row[1].set_xlim(left=0)
        for axis in row:
            axis.set_yscale("log")
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        r"M4 top-$k$ reward diagnostics: distance, normalized distance, and $v_t$",
        y=1.01,
    )
    fig.tight_layout()
    output = figure_path(
        "lr-ablation",
        "topk-distance-and-absorption",
        "step-and-effective-step",
        "runs-m4",
        "diagnostics",
    )
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(plot_topk_diagnostics())
