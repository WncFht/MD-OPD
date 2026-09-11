"""Plot M4 top-k reward distance and geometric absorption diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plots.config.runs import M4_RUNS
from plots.core.metrics import effective_step, load_run_metrics
from plots.core.plotting import BASELINE_STEP, figure_path, folded_baseline, mark_baseline, plot_series, style_axis

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "plots" / "runs"
TOPK_COLUMN = "opd/top_k_reward_mean"


def plot_topk_diagnostics() -> Path:
    """Render top-k distance and absorption on step/folded axes."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for run_id, spec in M4_RUNS.items():
        _, metrics = load_run_metrics(run_id, RUNS, column=TOPK_COLUMN)
        steps = metrics["step"].to_numpy(dtype=float)
        folded = effective_step(steps, spec["scale"])
        valid = np.isfinite(metrics["v_geo"])
        rows = [
            metrics["d_nats"].to_numpy(dtype=float),
            metrics["d_rel"].to_numpy(dtype=float),
            metrics["v_geo"].to_numpy(dtype=float) * 100,
        ]
        for row, y in enumerate(rows):
            mask = valid.to_numpy() if row == 2 else np.isfinite(y)
            plot_series(axes[row, 0], steps[mask], y[mask], color=spec["color"], linewidth=1.5, label=spec["label"])
            plot_series(
                axes[row, 1],
                folded[mask],
                y[mask],
                color=spec["color"],
                linewidth=1.5,
                label=spec["label"],
                baseline=effective_step(BASELINE_STEP, spec["scale"]),
            )
            folded_baseline(axes[row, 1], BASELINE_STEP, spec["scale"], spec["color"])

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
        row[0].grid(axis="y", alpha=0.25)
        row[0].spines[["top", "right"]].set_visible(False)
        style_axis(row[1])
        for axis in row:
            axis.set_yscale("log")
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
