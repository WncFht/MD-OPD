from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "alignment/runs"
OUT = ROOT / "alignment/figures"
RUNSPEC = [
    ("ybi7q89v", r"lr $5\times10^{-7}$", 0.5, "#b07a00"),
    ("mdxt04lm", r"lr $1\times10^{-6}$", 1.0, "#1f8a9a"),
    ("yj2w13fq", r"lr $2\times10^{-6}$", 2.0, "#245b9a"),
    ("tjjxnvw9", r"$\beta_2=0.95$", 1.0, "#7a3e9d"),
]


def data(rid):
    return (
        pd.read_csv(RUNS / f"{rid}__history.csv")
        .sort_values("_step")
        .reset_index(drop=True)
    )


def figure19(t0):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for rid, label, scale, color in RUNSPEC:
        d = data(rid)
        d = d[d._step >= t0]
        t = d._step.to_numpy()
        dist = -d["opd/token_reward_mean"].to_numpy()
        ax[1].plot(t, dist, color=color, lw=1.6, label=label)
        ax[2].plot(scale * t, dist / dist[0], color=color, lw=1.6, label=label)
        cols = [
            c for c in d if c.startswith("val-core/") and c.endswith("/acc/mean@16")
        ]
        if cols:
            v = d[cols].mean(axis=1)
            m = v.notna()
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
    ax[0].set_title("Average validation accuracy")
    ax[0].set_xlabel("training step $t$ (dimensionless)")
    ax[0].set_ylabel("mean@16 (3-task average)")
    ax[1].set_title(r"Absolute distance $d_t$")
    ax[1].set_xlabel("training step $t$ (dimensionless)")
    ax[1].set_ylabel("distance $d_t$ (nats/token)")
    ax[1].set_yscale("log")
    ax[2].set_title(rf"Normalised distance, collapsed ($t_0={t0}$)")
    ax[2].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    ax[2].set_ylabel(rf"normalised distance $d_t/d_{{{t0}}}$ (dimensionless)")
    ax[2].set_yscale("log")
    for a in ax:
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(OUT / f"figure19_m4_step{t0}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def d5(t0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rid, label, scale, color in RUNSPEC:
        d = data(rid)
        d = d[d._step >= t0]
        t = d._step.to_numpy()
        dist = -d["opd/token_reward_mean"].to_numpy()
        ax[0].plot(t, dist / dist[0], color=color, lw=1.7, label=label)
        ax[1].plot(scale * t, dist / dist[0], color=color, lw=1.7, label=label)
        ax[0].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[1].axvline(scale * t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
    ax[0].axvline(t0, color="black", ls=":", lw=1.2)
    ax[1].axvline(t0, color="black", ls=":", lw=1.2)
    ax[0].text(
        t0,
        ax[0].get_ylim()[1] * 0.8,
        f"analysis start t={t0}",
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax[1].text(
        t0,
        ax[1].get_ylim()[1] * 0.8,
        f"analysis start t={t0}",
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
    )
    for a in ax:
        a.set_yscale("log")
        a.set_ylabel(rf"normalised distance $d_t/d_{{{t0}}}$ (dimensionless)")
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(rf"M4 distance, training step ($t_0={t0}$)")
    ax[0].set_xlabel("training step $t$ (dimensionless)")
    ax[1].set_title(rf"M4 distance, folded axis ($t_0={t0}$)")
    ax[1].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"metric/D5_m4_distance_collapse_step{t0}.png", dpi=250)
    plt.close(fig)


def grad():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for rid, label, scale, color in RUNSPEC:
        d = data(rid)
        ax.plot(d._step, d["actor/grad_norm"], color=color, lw=1.5, label=label)
    ax.axhline(
        1.0, color="black", ls="--", lw=1, label="gradient-clip threshold $||g||=1$"
    )
    ax.axvline(60, color="grey", ls=":", lw=1, label="common baseline step 60")
    ax.set_yscale("log")
    ax.set_xlabel("training step $t$ (dimensionless)")
    ax.set_ylabel(r"pre-clip gradient norm $||g_t||_2$ (dimensionless)")
    ax.set_title("M4 gradient norm and clipping threshold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "metric/grad_norm_m4.png", dpi=250)
    plt.close(fig)


for t0 in (30, 60):
    figure19(t0)
    d5(t0)
grad()
