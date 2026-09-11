"""Plot local power-law exponents for sampled-token and top-k distances."""

from pathlib import Path

import matplotlib.pyplot as plt

from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.plotting import BASELINE_STEP, figure_path, mark_baseline, plot_series
from alignment.core.tail import local_alpha

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment" / "runs"


def lr_runs():
    """Return M4 learning-rate runs, excluding the beta2 ablation."""
    excluded = {"tjjxnvw9", "ybi7q89v"}
    return ((run_id, spec) for run_id, spec in M4_RUNS.items() if run_id not in excluded)


def alpha_series(run_id: str, column: str, baseline_step: int | None = None):
    """Compute local alpha from all positive steps in the complete history."""
    frame = load_run_history(run_id, RUNS).frame
    step = frame["_step"].to_numpy(dtype=float)
    distance = -frame[column].to_numpy(dtype=float)
    if baseline_step is not None:
        baseline = float(distance[step == baseline_step][0])
        distance = distance / baseline
    positive = step > 0
    return step[positive], local_alpha(step[positive], distance[positive])


def plot_metric(column: str, metric_name: str, suffix: str, folded: bool = False) -> Path:
    """Render one estimator's alpha curves for all learning-rate runs."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for run_id, spec in lr_runs():
        step, alpha = alpha_series(run_id, column)
        if folded:
            x = step * spec["scale"]
            plot_series(
                ax,
                x,
                alpha,
                color=spec["color"],
                linewidth=1.8,
                label=spec["label"],
                baseline=BASELINE_STEP * spec["scale"],
            )
            ax.axvline(BASELINE_STEP * spec["scale"], color=spec["color"], ls=":", lw=0.7, alpha=0.5)
        else:
            plot_series(ax, step, alpha, color=spec["color"], linewidth=1.8, label=spec["label"])
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, label=r"$\alpha=1$")
    if folded:
        ax.set_xlim(left=0)
    else:
        mark_baseline(ax)
    ax.set_xlabel(r"effective time $u=(lr/10^{-6})t$" if folded else "training step $t$")
    ax.set_ylabel(r"local exponent $\alpha(t)=-d\log d/d\log t$")
    ax.set_title(rf"M4 local exponent $\alpha(t)$: {metric_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    axis_name = "effective-step" if folded else "step"
    output = figure_path("lr-ablation", "local-alpha", axis_name, "runs-m4", suffix)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_normalized_folded(column: str, metric_name: str, suffix: str) -> Path:
    """Render alpha with the pre-baseline segment faded at each run's fold."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for run_id, spec in lr_runs():
        step, alpha = alpha_series(run_id, column, baseline_step=BASELINE_STEP)
        x = step * spec["scale"]
        baseline_x = BASELINE_STEP * spec["scale"]
        plot_series(ax, x, alpha, color=spec["color"], linewidth=1.8, label=spec["label"], baseline=baseline_x)
        ax.axvline(baseline_x, color=spec["color"], ls="--", lw=0.8, alpha=0.7)
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, label=r"$\alpha=1$")
    ax.set_xlim(left=0)
    ax.set_xlabel(r"effective time $u=(lr/10^{-6})t$")
    ax.set_ylabel(r"local exponent $\alpha(t)$")
    ax.set_title(rf"M4 $\alpha(t)$: {metric_name}; step-30 fold baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    output = figure_path("lr-ablation", "local-alpha-normalized", "effective-step", "runs-m4", suffix)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    """Render separate sampled-token and top-k alpha figures."""
    print(plot_metric("opd/token_reward_mean", "sampled-token KL proxy", "sampled-token"))
    print(plot_metric("opd/top_k_reward_mean", r"top-$k$ reward", "top-k"))
    print(
        plot_metric(
            "opd/token_reward_mean",
            "sampled-token KL proxy",
            "sampled-token-folded",
            folded=True,
        )
    )
    print(
        plot_normalized_folded(
            "opd/token_reward_mean",
            "sampled-token KL proxy",
            "sampled-token-folded-step30",
        )
    )
    print(
        plot_normalized_folded(
            "opd/top_k_reward_mean",
            r"top-$k$ reward",
            "top-k-folded-step30",
        )
    )
    print(
        plot_metric(
            "opd/top_k_reward_mean",
            r"top-$k$ reward",
            "top-k-folded",
            folded=True,
        )
    )


if __name__ == "__main__":
    main()
