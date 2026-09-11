from pathlib import Path

import matplotlib.pyplot as plt

from plots.config.runs import M4_RUNS
from plots.core.accuracy import accuracy_ylim, add_accuracy_baselines
from plots.core.metrics import effective_step, load_run_metrics, normalize_at_step, validation_mean
from plots.core.plotting import BASELINE_STEP, figure_path, folded_baseline, mark_baseline, plot_series, style_axis

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "plots/runs"

# row -> normalization baseline step for the six-panel distance figure
ROW_BASELINES = [BASELINE_STEP, BASELINE_STEP, 60]


def main():
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex="col")
    for rid, spec in M4_RUNS.items():
        label, scale, color = spec["label"], spec["scale"], spec["color"]
        _, metrics = load_run_metrics(rid, RUNS)
        t = metrics["step"].to_numpy(float)
        dist = metrics["d_nats"].to_numpy(float)
        u = effective_step(t, scale)
        vals = [dist, metrics["d_rel"].to_numpy(float), normalize_at_step(t, dist, 60)]
        for row, y in enumerate(vals):
            base = ROW_BASELINES[row]
            if row == 0:
                axs[row, 0].plot(t, y, color=color, lw=1.6)
                axs[row, 1].plot(u, y, color=color, lw=1.6)
            else:
                plot_series(axs[row, 0], t, y, color=color, baseline=base, linewidth=1.6)
                plot_series(axs[row, 1], u, y, color=color, baseline=effective_step(base, scale), linewidth=1.6)
            folded_baseline(axs[row, 1], base, scale, color)
        axs[0, 0].plot([], [], color=color, lw=1.6, label=label)
    for r, base in enumerate(ROW_BASELINES):
        mark_baseline(axs[r, 0], base)
        style_axis(axs[r, 1])
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
    observed = []
    for rid, spec in M4_RUNS.items():
        history, _ = load_run_metrics(rid, RUNS)
        v = validation_mean(history.frame)
        m = v.notna()
        observed.append(v[m].to_numpy() * 100)
        ax.plot(
            history.frame.loc[m, "_step"],
            v[m] * 100,
            "-s",
            ms=3,
            lw=1.4,
            color=spec["color"],
            label=spec["label"],
        )
    add_accuracy_baselines(ax, as_percent=True)
    mark_baseline(ax)
    ax.set_ylim(accuracy_ylim(observed))
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
