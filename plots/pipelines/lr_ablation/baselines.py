from pathlib import Path

import matplotlib.pyplot as plt

from plots.config.runs import M4_RUNS
from plots.core.accuracy import accuracy_ylim, add_accuracy_baselines
from plots.core.metrics import effective_step, load_run_metrics, normalize_at_step, validation_mean
from plots.core.plotting import BASELINE_STEP, figure_path, folded_baseline, mark_baseline, plot_series, style_axis

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "plots/runs"
# figure19 excludes the beta2 arm; d5 and grad cover the full M4_RUNS set
FIG19_RUNS = {rid: spec for rid, spec in M4_RUNS.items() if rid != "tjjxnvw9"}


def collapse(t0: int | None, runs=M4_RUNS):
    """Validation, absolute distance, and lr-folded distance panels.

    ``t0=None`` draws the absolute distance on the folded axis (the figure-19
    variant, restricted to the lr arms); a step value draws ``d_t/d_{t0}``
    with the matching fade.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    observed = []
    for rid, spec in runs.items():
        label, scale, color = spec["label"], spec["scale"], spec["color"]
        history, metrics = load_run_metrics(rid, RUNS)
        t = metrics["step"].to_numpy()
        dist = metrics["d_nats"].to_numpy()
        folded = effective_step(t, scale)
        folded_base = effective_step(BASELINE_STEP, scale)
        plot_series(ax[1], t, dist, color=color, linewidth=1.6, label=label, log_scale=True)
        folded_dist = dist if t0 is None else normalize_at_step(t, dist, t0)
        plot_series(
            ax[2],
            folded,
            folded_dist,
            color=color,
            linewidth=1.6,
            label=label,
            baseline=folded_base,
            log_scale=True,
        )
        v = validation_mean(history.frame)
        m = v.notna().to_numpy()
        observed.append(v[m].to_numpy() * 100)
        ax[0].plot(t[m], v[m] * 100, "-s", ms=3, lw=1.2, color=color, label=label)
        for axis, x in ((ax[1], t[-1]), (ax[2], folded[-1])):
            axis.axvline(x, color=color, ls="--", lw=0.8, alpha=0.65)
        folded_baseline(ax[2], BASELINE_STEP, scale, color)
    add_accuracy_baselines(ax[0], as_percent=True)
    ax[0].set_ylim(accuracy_ylim(observed))
    mark_baseline(ax[0])
    mark_baseline(ax[1])
    style_axis(ax[2])
    ax[0].set_title("Average validation accuracy")
    ax[0].set_xlabel("training step $t$ (dimensionless)")
    ax[0].set_ylabel("mean@16 (%, 3-task average)")
    ax[1].set_title(r"Absolute distance $d_t$")
    ax[1].set_xlabel("training step $t$ (dimensionless)")
    ax[1].set_ylabel("distance $d_t$ (nats/token)")
    if t0 is None:
        ax[2].set_title(r"Absolute Distance $d_t$ (Collapsed)")
        ax[2].set_ylabel(r"distance $d_t$ (nats/token)")
        variant = "figure19"
    else:
        ax[2].set_title(rf"Normalised distance, collapsed ($t_0={t0}$)")
        ax[2].set_ylabel(rf"normalised distance $d_t/d_{{{t0}}}$ (dimensionless)")
        variant = f"norm-d{t0}"
    ax[2].set_xlabel(r"effective time $u=(lr/10^{-6})t$ (dimensionless)")
    for a in ax:
        a.grid(axis="y", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(
        figure_path("lr-ablation", "validation-distance-collapse", "step", "runs-m4", variant),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def d5(t0):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rid, spec in M4_RUNS.items():
        label, scale, color = spec["label"], spec["scale"], spec["color"]
        _, metrics = load_run_metrics(rid, RUNS)
        t = metrics.step.to_numpy()
        dist = normalize_at_step(t, metrics.d_nats.to_numpy(), t0)
        folded = effective_step(t, scale)
        plot_series(ax[0], t, dist, color=color, linewidth=1.7, label=label, baseline=t0, log_scale=True)
        plot_series(
            ax[1],
            folded,
            dist,
            color=color,
            linewidth=1.7,
            label=label,
            baseline=effective_step(t0, scale),
            log_scale=True,
        )
        ax[0].axvline(t[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        ax[1].axvline(folded[-1], color=color, ls="--", lw=0.8, alpha=0.65)
        folded_baseline(ax[1], t0, scale, color)
    mark_baseline(ax[0], t0)
    style_axis(ax[1])
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
    for rid, spec in M4_RUNS.items():
        history, metrics = load_run_metrics(rid, RUNS)
        ax.plot(metrics.step, history.frame["actor/grad_norm"], color=spec["color"], lw=1.5, label=spec["label"])
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
    collapse(None, FIG19_RUNS)
    for t0 in (30, 60):
        collapse(t0)
        d5(t0)
    grad()


if __name__ == "__main__":
    main()
