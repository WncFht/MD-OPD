from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alignment.config.runs import M4_RUNS
from alignment.core.accuracy import add_accuracy_baselines
from alignment.core.data import load_run_history
from alignment.core.metrics import (
    compute_alignment_metrics,
    effective_step,
    validation_mean,
)
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment/runs"
RUNSPEC = [(rid, spec["label"], spec["scale"], spec["color"]) for rid, spec in M4_RUNS.items()]

# row -> normalization baseline step for the six-panel distance figure
ROW_BASELINES = [BASELINE_STEP, BASELINE_STEP, 60]


def load(rid):
    history = load_run_history(rid, RUNS)
    return history.frame, compute_alignment_metrics(history)


def main():
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for rid, label, scale, color in RUNSPEC:
        d, metrics = load(rid)
        t = metrics["step"].to_numpy(float)
        dist = metrics["d_nats"].to_numpy(float)
        u = effective_step(t, scale)
        vals = [dist, dist / np.interp(30, t, dist), dist / np.interp(60, t, dist)]
        for row, y in enumerate(vals):
            base = ROW_BASELINES[row]
            if row == 0:
                axs[row, 0].plot(t, y, color=color, lw=1.6)
                axs[row, 1].plot(u, y, color=color, lw=1.6)
            else:
                plot_series(axs[row, 0], t, y, color=color, baseline=base, linewidth=1.6)
                plot_series(axs[row, 1], u, y, color=color, baseline=scale * base, linewidth=1.6)
            axs[row, 1].axvline(scale * base, color=color, ls="--", lw=0.7, alpha=0.5)
        axs[0, 0].plot([], [], color=color, lw=1.6, label=label)
    for r, base in enumerate(ROW_BASELINES):
        mark_baseline(axs[r, 0], base)
        axs[r, 1].set_xlim(left=0)
        for c in range(2):
            axs[r, c].set_yscale("log")
            axs[r, c].grid(axis="y", alpha=0.25)
            axs[r, c].spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_title("Absolute distance on training step")
    axs[0, 1].set_title("Absolute distance on folded time")
    axs[1, 0].set_title(r"Normalised distance $d_t/d_{30}$ on training step")
    axs[1, 1].set_title(r"Normalised distance $d_t/d_{30}$ on folded time")
    axs[2, 0].set_title(r"Normalised distance $d_t/d_{60}$ on training step")
    axs[2, 1].set_title(r"Normalised distance $d_t/d_{60}$ on folded time")
    for r in range(3):
        axs[r, 0].set_ylabel(
            [
                "distance $d_t$ (nats/token)",
                r"$d_t/d_{30}$ (dimensionless)",
                r"$d_t/d_{60}$ (dimensionless)",
            ][r]
        )
    axs[2, 0].set_xlabel("training step $t$ (dimensionless)")
    axs[2, 1].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    axs[0, 0].text(30, axs[0, 0].get_ylim()[1] * 0.7, "30", rotation=90, fontsize=8, color="grey")
    axs[0, 0].text(60, axs[0, 0].get_ylim()[1] * 0.7, "60", rotation=90, fontsize=8, color="grey")
    axs[0, 0].axvline(60, color="grey", ls=":", lw=0.9)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("M4 distance diagnostics: absolute and two normalisation baselines", y=1.03)
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "distance", "step-and-effective-step", "runs-m4", "six-panel"),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def accuracy():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for rid, label, scale, color in RUNSPEC:
        d, _ = load(rid)
        v = validation_mean(d)
        m = v.notna()
        ax.plot(d.loc[m, "_step"], v[m] * 100, "-s", ms=3, lw=1.4, color=color, label=label)
    add_accuracy_baselines(ax, as_percent=True)
    mark_baseline(ax)
    ax.set_xlabel("training step $t$ (dimensionless)")
    ax.set_ylabel("validation accuracy (%, 3-task mean@16)")
    ax.set_title("M4 validation accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "validation-accuracy", "step", "runs-m4", "mean16"),
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
    accuracy()
