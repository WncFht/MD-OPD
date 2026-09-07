from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parent / "runs"
O = Path(__file__).resolve().parent / "figures" / "metric"
O.mkdir(exist_ok=True)
RUNS = [
    ("mdxt04lm", "lr 1e-6", "#1f8a9a", 1),
    ("tjjxnvw9", r"$\beta_2=0.95$", "#7a3e9d", 1),
    ("yj2w13fq", "lr 2e-6", "#245b9a", 2),
    ("ybi7q89v", "lr 5e-7", "#b07a00", 0.5),
]


def load(r):
    return pd.read_csv(R / f"{r}__history.csv").sort_values("_step")


def d1():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rid, l, c, s in RUNS:
        x = load(rid)
        t = x["_step"]
        ce = -x["opd/teacher_logp_mean"]
        h = x["actor/entropy"]
        k = -x["opd/token_reward_mean"]
        for y, ls, lw in [(ce, "-", 1.5), (h, ":", 1.0), (k, "--", 1.7)]:
            axs[0].plot(t[t < 60], y[t < 60], color=c, lw=lw, ls=ls, alpha=0.25)
            axs[0].plot(t[t >= 60], y[t >= 60], color=c, lw=lw, ls=ls)
        ratio = 100 * k / ce
        axs[1].plot(t[t < 60], ratio[t < 60], color=c, lw=1.2, alpha=0.25)
        axs[1].plot(t[t >= 60], ratio[t >= 60], color=c, lw=1.7, label=l)
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
    fig.savefig(O / "D1_multirun_ce_entropy_kl.png", dpi=250)
    plt.close(fig)


def d2():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for rid, l, c, s in RUNS:
        x = load(rid)
        t = x["_step"]
        k = -x["opd/token_reward_mean"]
        base = k.iloc[(t - 60).abs().argmin()]
        axs[0].plot(t[t < 60], (k / base)[t < 60], color=c, lw=1.2, alpha=0.25)
        axs[0].plot(t[t >= 60], (k / base)[t >= 60], color=c, lw=1.8, label=l)
        axs[1].plot(t, k, color=c, lw=1.8, label=l)
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
    fig.savefig(O / "D2_multirun_mean_collapse.png", dpi=250)
    plt.close(fig)


def d5():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rid, l, c, s in RUNS:
        x = load(rid)
        t = x["_step"].to_numpy()
        k = -x["opd/token_reward_mean"].to_numpy()
        axs[0].plot(t, k / k[0], color=c, lw=1.7, label=l)
        axs[1].plot(s * t, k / k[0], color=c, lw=1.7, label=l)
        axs[0].axvline(t[-1], color=c, ls="--", lw=0.8, alpha=0.6)
        axs[1].axvline(s * t[-1], color=c, ls="--", lw=0.8, alpha=0.6)
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
    fig.savefig(O / "D5_m4_distance_collapse.png", dpi=250)
    plt.close(fig)


d1()
d2()
d5()
