from pathlib import Path

import matplotlib.pyplot as plt

from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.metrics import (
    ce_entropy_kl,
    compute_alignment_metrics,
    effective_step,
)
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series

R = Path(__file__).resolve().parents[3] / "alignment" / "runs"
RUNS = [(rid, spec["label"], spec["color"], spec["scale"]) for rid, spec in M4_RUNS.items()]


def load(r):
    history = load_run_history(r, R)
    return history.frame, compute_alignment_metrics(history)


def d1():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rid, label, color, scale in RUNS:
        x, metrics = load(rid)
        t = metrics["step"]
        decomposition = ce_entropy_kl(x)
        ce = decomposition["ce"]
        h = decomposition["entropy"]
        k = decomposition["kl_proxy"]
        for y, ls, lw in [(ce, "-", 1.5), (h, ":", 1.0), (k, "--", 1.7)]:
            plot_series(axs[0], t, y, color=color, linewidth=lw, linestyle=ls, baseline=60)
        ratio = 100 * k / ce
        plot_series(axs[1], t, ratio, color=color, linewidth=1.7, label=label, baseline=60, log_scale=True)
    for a in axs:
        mark_baseline(a, 60)
        a.axvline(BASELINE_STEP, color="grey", ls=":", lw=0.8)
    axs[0].set_yscale("log")
    axs[0].set_title("CE, entropy, and KL proxy across M4")
    axs[0].set_ylabel("nats/token")
    axs[0].set_xlabel("step")
    axs[0].text(
        0.02,
        0.03,
        "solid=CE, dotted=H, dashed=KL",
        transform=axs[0].transAxes,
        fontsize=8,
    )
    axs[1].set_title("KL proxy as fraction of CE")
    axs[1].set_ylabel("100·KL/CE (%)")
    axs[1].set_xlabel("step")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_path("lr-ablation", "ce-entropy-kl", "step", "runs-m4", "decomposition"), dpi=250)
    plt.close(fig)


def d2():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for rid, label, color, scale in RUNS:
        x, metrics = load(rid)
        t = metrics["step"]
        k = metrics["d_nats"]
        base = k.iloc[(t - 60).abs().argmin()]
        plot_series(axs[0], t, k / base, color=color, linewidth=1.8, label=label, baseline=60, log_scale=True)
        plot_series(axs[1], t, k, color=color, linewidth=1.8, label=label, baseline=60, log_scale=True)
    for a in axs:
        mark_baseline(a, 60)
        a.axvline(BASELINE_STEP, color="grey", ls=":", lw=0.8)
    axs[0].set_title("Mean distance collapse (normalised at step 60)")
    axs[0].set_ylabel(r"normalised distance $d_t/d_{60}$ (dimensionless)")
    axs[1].set_title("Mean KL proxy (absolute)")
    axs[1].set_ylabel(r"distance $d_t$ (nats/token)")
    for a in axs:
        a.set_xlabel("training step $t$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "distance", "step", "runs-m4", "norm-d60-and-absolute"),
        dpi=250,
    )
    plt.close(fig)


def d5():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rid, label, color, scale in RUNS:
        x, metrics = load(rid)
        t = metrics["step"].to_numpy()
        k = metrics["d_rel"].to_numpy()
        plot_series(axs[0], t, k, color=color, linewidth=1.7, label=label, log_scale=True)
        plot_series(
            axs[1],
            effective_step(t, scale),
            k,
            color=color,
            linewidth=1.7,
            label=label,
            baseline=BASELINE_STEP * scale,
            log_scale=True,
        )
        axs[0].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.6)
        axs[1].axvline(scale * t[-1], color=color, ls="--", lw=0.8, alpha=0.6)
        axs[1].axvline(scale * BASELINE_STEP, color=color, ls=":", lw=0.7, alpha=0.5)
    mark_baseline(axs[0])
    axs[0].set_title("M4 distance on training-step axis")
    axs[1].set_title("M4 distance on folded axis")
    axs[0].set_xlabel("training step $t$ (dimensionless)")
    axs[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    axs[1].set_xlim(left=0)
    for a in axs:
        a.set_ylabel(r"normalised distance $d_t/d_{30}$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("lr-ablation", "distance", "effective-step", "runs-m4", "norm-d30"), dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    d1()
    d2()
    d5()
