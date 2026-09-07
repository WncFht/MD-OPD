from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config.runs import M4_RUNS
from core.data import load_run_history
from core.metrics import compute_alignment_metrics, effective_step, validation_mean

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "alignment/runs"
OUT = ROOT / "alignment/figures"
RUNSPEC = [(rid, spec["label"], spec["scale"], spec["color"]) for rid, spec in M4_RUNS.items()]


def load(rid):
    history = load_run_history(rid, RUNS)
    return history.frame, compute_alignment_metrics(history)


def split_plot(ax, x, y, color, cut):
    m = x < cut
    ax.plot(x[m], y[m], color=color, alpha=0.25, lw=1.2)
    ax.plot(x[~m], y[~m], color=color, lw=1.6)


def main():
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for rid, label, scale, color in RUNSPEC:
        d, metrics = load(rid)
        t = metrics["step"].to_numpy(float)
        dist = metrics["d_nats"].to_numpy(float)
        u = effective_step(t, scale)
        vals = [dist, dist / np.interp(30, t, dist), dist / np.interp(60, t, dist)]
        for row, y in enumerate(vals):
            if row == 0:
                axs[row, 0].plot(t, y, color=color, lw=1.6)
                axs[row, 1].plot(u, y, color=color, lw=1.6)
            else:
                cut = 30 if row == 1 else 60
                split_plot(axs[row, 0], t, y, color, cut)
                split_plot(axs[row, 1], u, y, color, scale * cut)
            axs[row, 0].axvline(30, color="grey", ls="--", lw=0.8)
            axs[row, 0].axvline(60, color="grey", ls=":", lw=0.9)
            axs[row, 1].axvline(scale * 30, color="grey", ls="--", lw=0.8)
            axs[row, 1].axvline(scale * 60, color="grey", ls=":", lw=0.9)
        axs[0, 0].plot([], [], color=color, lw=1.6, label=label)
    for r in range(3):
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
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("M4 distance diagnostics: absolute and two normalisation baselines", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "metric/m4_distance_six_panel.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def accuracy():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for rid, label, scale, color in RUNSPEC:
        d, _ = load(rid)
        cols = [c for c in d if c.startswith("val-core/") and c.endswith("/acc/mean@16")]
        if not cols:
            continue
        v = validation_mean(d)
        m = v.notna()
        ax.plot(d.loc[m, "_step"], v[m] * 100, "-s", ms=3, lw=1.4, color=color, label=label)
    ax.set_xlabel("training step $t$ (dimensionless)")
    ax.set_ylabel("validation accuracy (%, 3-task mean@16)")
    ax.set_title("M4 validation accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "metric/m4_validation_accuracy.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
    accuracy()
