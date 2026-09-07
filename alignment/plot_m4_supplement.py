from pathlib import Path

import matplotlib.pyplot as plt
from config.runs import M4_RUNS
from core.data import load_run_history
from core.metrics import ce_entropy_kl, compute_alignment_metrics, effective_step

R = Path(__file__).resolve().parent / "runs"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "metric"
OUTPUT_DIR.mkdir(exist_ok=True)
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
        t = metrics["step"]
        ce = decomposition["ce"]
        h = decomposition["entropy"]
        k = decomposition["kl_proxy"]
        for y, ls, lw in [(ce, "-", 1.5), (h, ":", 1.0), (k, "--", 1.7)]:
            axs[0].plot(t[t < 60], y[t < 60], color=color, lw=lw, ls=ls, alpha=0.25)
            axs[0].plot(t[t >= 60], y[t >= 60], color=color, lw=lw, ls=ls)
        ratio = 100 * k / ce
        axs[1].plot(t[t < 60], ratio[t < 60], color=color, lw=1.2, alpha=0.25)
        axs[1].plot(t[t >= 60], ratio[t >= 60], color=color, lw=1.7, label=label)
    for a in axs:
        a.axvline(30, color="grey", ls="--", lw=0.8)
        a.axvline(60, color="grey", ls=":", lw=0.9)
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
    axs[1].set_yscale("log")
    axs[1].set_title("KL proxy as fraction of CE")
    axs[1].set_ylabel("100·KL/CE (%)")
    axs[1].set_xlabel("step")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "D1_multirun_ce_entropy_kl.png", dpi=250)
    plt.close(fig)


def d2():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for rid, label, color, scale in RUNS:
        x, metrics = load(rid)
        t = metrics["step"]
        k = metrics["d_nats"]
        base = k.iloc[(t - 60).abs().argmin()]
        axs[0].plot(t[t < 60], (k / base)[t < 60], color=color, lw=1.2, alpha=0.25)
        axs[0].plot(t[t >= 60], (k / base)[t >= 60], color=color, lw=1.8, label=label)
        axs[1].plot(t, k, color=color, lw=1.8, label=label)
    for a in axs:
        a.axvline(30, color="grey", ls="--", lw=0.8)
        a.axvline(60, color="grey", ls=":", lw=0.9)
    axs[0].set_yscale("log")
    axs[0].set_title("Mean distance collapse (normalised at step 60)")
    axs[0].set_ylabel(r"normalised distance $d_t/d_{60}$ (dimensionless)")
    axs[1].set_yscale("log")
    axs[1].set_title("Mean KL proxy (absolute)")
    axs[1].set_ylabel(r"distance $d_t$ (nats/token)")
    for a in axs:
        a.set_xlabel("training step $t$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "D2_multirun_mean_collapse.png", dpi=250)
    plt.close(fig)


def d5():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rid, label, color, scale in RUNS:
        x, metrics = load(rid)
        t = metrics["step"].to_numpy()
        k = metrics["d_nats"].to_numpy()
        axs[0].plot(t, k / k[0], color=color, lw=1.7, label=label)
        axs[1].plot(
            effective_step(t.to_numpy(), scale),
            k / k[0],
            color=color,
            lw=1.7,
            label=label,
        )
        axs[0].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.6)
        axs[1].axvline(scale * t[-1], color=color, ls="--", lw=0.8, alpha=0.6)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_title("M4 distance on training-step axis")
    axs[1].set_title("M4 distance on folded axis")
    axs[0].set_xlabel("training step $t$ (dimensionless)")
    axs[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    for a in axs:
        a.set_ylabel(r"normalised distance $d_t/d_{30}$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "D5_m4_distance_collapse.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    d1()
    d2()
    d5()
