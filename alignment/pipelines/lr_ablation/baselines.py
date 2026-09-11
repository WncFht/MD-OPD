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
from alignment.core.plotting import figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment/runs"
RUNSPEC = [(rid, settings["label"], settings["scale"], settings["color"]) for rid, settings in M4_RUNS.items()]


def data(rid):
    history = load_run_history(rid, RUNS)
    return history.frame, compute_alignment_metrics(history)


def figure19(t0):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for rid, label, scale, color in RUNSPEC:
        raw, metrics = data(rid)
        t = metrics.step.to_numpy()
        dist = metrics.d_nats.to_numpy()
        plot_series(ax[1], t, dist, color=color, linewidth=1.6, label=label, baseline=t0, log_scale=True)
        plot_series(
            ax[2],
            effective_step(t, scale),
            dist / np.interp(t0, t, dist),
            color=color,
            linewidth=1.6,
            label=label,
            baseline=scale * t0,
            log_scale=True,
        )
        v = validation_mean(raw)
        m = v.notna().to_numpy()
        ax[0].plot(t[m], v[m], "-s", ms=3, lw=1.2, color=color, label=label)
        ax[1].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[1].text(
            t[-1],
            ax[1].get_ylim()[1] * 0.75,
            f"{label}: {int(t[-1])}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=color,
        )
        ax[2].axvline(scale * t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[2].text(
            scale * t[-1],
            ax[2].get_ylim()[1] * 0.75,
            f"{label}: u={scale * t[-1]:.0f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=color,
        )
    add_accuracy_baselines(ax[0], as_percent=False)
    mark_baseline(ax[0], t0)
    mark_baseline(ax[1], t0)
    ax[2].set_xlim(left=0)
    ax[0].set_title("Average validation accuracy")
    ax[0].set_xlabel("training step $t$ (dimensionless)")
    ax[0].set_ylabel("mean@16 (3-task average)")
    ax[1].set_title(r"Absolute distance $d_t$")
    ax[1].set_xlabel("training step $t$ (dimensionless)")
    ax[1].set_ylabel("distance $d_t$ (nats/token)")
    ax[2].set_title(rf"Normalised distance, collapsed ($t_0={t0}$)")
    ax[2].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    ax[2].set_ylabel(rf"normalised distance $d_t/d_{{{t0}}}$ (dimensionless)")
    for a in ax:
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "validation-distance-collapse", "step", "runs-m4", f"norm-d{t0}"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def d5(t0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rid, label, scale, color in RUNSPEC:
        _, metrics = data(rid)
        t = metrics.step.to_numpy()
        dist = metrics.d_nats.to_numpy() / np.interp(t0, metrics.step.to_numpy(), metrics.d_nats.to_numpy())
        plot_series(ax[0], t, dist, color=color, linewidth=1.7, label=label, baseline=t0, log_scale=True)
        plot_series(
            ax[1],
            effective_step(t, scale),
            dist,
            color=color,
            linewidth=1.7,
            label=label,
            baseline=scale * t0,
            log_scale=True,
        )
        ax[0].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[1].axvline(scale * t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[1].axvline(scale * t0, color=color, ls=":", lw=0.7, alpha=0.5)
    mark_baseline(ax[0], t0)
    ax[1].set_xlim(left=0)
    for a in ax:
        a.set_ylabel(rf"normalised distance $d_t/d_{{{t0}}}$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(rf"M4 distance, training step ($t_0={t0}$)")
    ax[0].set_xlabel("training step $t$ (dimensionless)")
    ax[1].set_title(rf"M4 distance, folded axis ($t_0={t0}$)")
    ax[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "distance", "effective-step", "runs-m4", f"norm-d{t0}"),
        dpi=250,
    )
    plt.close(fig)


def grad():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for rid, label, scale, color in RUNSPEC:
        d, metrics = data(rid)
        ax.plot(metrics.step, d["actor/grad_norm"], color=color, lw=1.5, label=label)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="gradient-clip threshold $||g||=1$")
    mark_baseline(ax, 60)
    ax.plot([], [], color="#555555", ls="--", lw=0.9, label="common baseline step 60")
    ax.set_yscale("log")
    ax.set_xlabel("training step $t$ (dimensionless)")
    ax.set_ylabel(r"pre-clip gradient norm $||g_t||_2$ (dimensionless)")
    ax.set_title("M4 gradient norm and clipping threshold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "gradient-norm", "step", "runs-m4", "optimizer-clipping"),
        dpi=250,
    )
    plt.close(fig)


def main():
    for t0 in (30, 60):
        figure19(t0)
        d5(t0)
    grad()


if __name__ == "__main__":
    main()
