"""Plot the M4 geometric-window absorption rate ``v_t``."""

from pathlib import Path

import matplotlib.pyplot as plt

from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.metrics import compute_alignment_metrics, effective_step
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment" / "runs"


def plot_absorption_rate() -> Path:
    """Render M4 geometric-window absorption rates on step and folded axes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for run_id, spec in M4_RUNS.items():
        history = load_run_history(run_id, RUNS)
        metrics = compute_alignment_metrics(history)
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
            baseline=BASELINE_STEP * spec["scale"],
        )
        axes[1].axvline(BASELINE_STEP * spec["scale"], color=spec["color"], ls=":", lw=0.7, alpha=0.5)

    axes[0].set_xlabel("training step $t$")
    axes[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$")
    axes[0].set_ylabel(r"geometric-window absorption rate $v_t$ (% per step)")
    axes[0].set_title(r"$v_t$ on training-step axis")
    axes[1].set_title(r"$v_t$ on folded effective-time axis")
    mark_baseline(axes[0])
    axes[1].set_xlim(left=0)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(r"M4 absorption rate $v_t$ (geometric window $[t/1.4,1.4t]$)", y=1.02)
    fig.tight_layout()
    output = figure_path("lr-ablation", "absorption-rate", "step-and-effective-step", "runs-m4", "geometric")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(plot_absorption_rate())
