#!/usr/bin/env python3
"""Figure-19-style M4 comparison: validation, distance, and lr-collapsed distance."""

from pathlib import Path

import matplotlib.pyplot as plt

from alignment.config.runs import M4_RUNS
from alignment.core.accuracy import add_accuracy_baselines
from alignment.core.data import load_run_history
from alignment.core.metrics import compute_alignment_metrics, validation_mean
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment/runs"
OUT = figure_path("lr-ablation", "validation-distance-collapse", "step", "runs-m4", "figure19")


def main() -> Path:
    """Render validation accuracy, absolute distance, and lr-folded distance."""
    spec = [
        (rid, settings["label"], settings["scale"], settings["color"])
        for rid, settings in M4_RUNS.items()
        if rid != "tjjxnvw9"
    ]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for rid, label, scale, color in spec:
        history = load_run_history(rid, RUNS)
        d = history.frame
        metrics = compute_alignment_metrics(history)
        t = metrics["step"].to_numpy()
        dist = metrics["d_nats"].to_numpy()
        plot_series(ax[1], t, dist, color=color, linewidth=1.7, label=label, log_scale=True)
        plot_series(
            ax[2],
            scale * t,
            dist,
            color=color,
            linewidth=1.7,
            label=label,
            baseline=BASELINE_STEP * scale,
            log_scale=True,
        )
        vv = validation_mean(d)
        m = vv.notna()
        ax[0].plot(t[m], vv[m], color=color, marker="s", ms=3, lw=1.4, label=label)
        ax[2].axvline(scale * BASELINE_STEP, color=color, ls=":", lw=0.7, alpha=0.5)
    add_accuracy_baselines(ax[0], as_percent=False)
    mark_baseline(ax[0])
    mark_baseline(ax[1])
    ax[2].set_xlim(left=0)
    ax[0].set_title("Average Validation Accuracy")
    ax[0].set_xlabel("Training Step")
    ax[0].set_ylabel("mean@16 (3-task average)")
    ax[1].set_title(r"Distance $d_t$")
    ax[1].set_xlabel("Training Step $t$")
    ax[1].set_ylabel("nats/token")
    ax[2].set_title(r"Absolute Distance $d_t$ (Collapsed)")
    ax[2].set_xlabel(r"Effective Time $u=(lr/10^{-6})t$")
    ax[2].set_ylabel(r"distance $d_t$ (nats/token)")
    for a in ax:
        a.grid(True, axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
