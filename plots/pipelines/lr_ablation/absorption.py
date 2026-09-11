"""Plot the M4 geometric-window absorption rate ``v_t``."""

from pathlib import Path

import matplotlib.pyplot as plt

from plots.config.runs import M4_RUNS
from plots.core.metrics import effective_step, load_run_metrics
from plots.core.plotting import BASELINE_STEP, figure_path, folded_baseline, mark_baseline, plot_series, style_axis

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "plots" / "runs"


def plot_absorption_rate() -> Path:
    """Render M4 geometric-window absorption rates on step and folded axes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for run_id, spec in M4_RUNS.items():
        _, metrics = load_run_metrics(run_id, RUNS)
        mask = metrics["v_geo"].notna()
        step = metrics.loc[mask, "step"].to_numpy(float)
        rate = metrics.loc[mask, "v_geo"].to_numpy(float) * 100
        folded = effective_step(step, spec["scale"])
        plot_series(axes[0], step, rate, color=spec["color"], linewidth=1.8, label=spec["label"])
        plot_series(
            axes[1],
            folded,
            rate,
            color=spec["color"],
            linewidth=1.8,
            label=spec["label"],
            baseline=effective_step(BASELINE_STEP, spec["scale"]),
        )
        folded_baseline(axes[1], BASELINE_STEP, spec["scale"], spec["color"])

    axes[0].set_xlabel("training step $t$")
    axes[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$")
    axes[0].set_ylabel(r"geometric-window absorption rate $v_t$ (% per step)")
    axes[0].set_title(r"$v_t$ on training-step axis")
    axes[1].set_title(r"$v_t$ on folded effective-time axis")
    mark_baseline(axes[0])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    style_axis(axes[1])
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(r"M4 absorption rate $v_t$ (geometric window $[t/1.4,1.4t]$)", y=1.02)
    fig.tight_layout()
    output = figure_path("lr-ablation", "absorption-rate", "step-and-effective-step", "runs-m4", "geometric")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(plot_absorption_rate())
