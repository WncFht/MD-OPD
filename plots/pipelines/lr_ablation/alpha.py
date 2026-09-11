"""Plot local power-law exponents for sampled-token and top-k distances."""

from pathlib import Path

import matplotlib.pyplot as plt

from plots.config.runs import M4_RUNS
from plots.core.metrics import effective_step, load_run_metrics
from plots.core.plotting import BASELINE_STEP, figure_path, folded_baseline, mark_baseline, plot_series, style_axis

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "plots" / "runs"


def lr_runs():
    """Return M4 learning-rate runs, excluding the beta2 ablation."""
    excluded = {"tjjxnvw9", "ybi7q89v"}
    return ((run_id, spec) for run_id, spec in M4_RUNS.items() if run_id not in excluded)


def plot_metric(column: str, metric_name: str, suffix: str, folded: bool = False) -> Path:
    """Render one estimator's alpha curves for all learning-rate runs."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for run_id, spec in lr_runs():
        _, metrics = load_run_metrics(run_id, RUNS, column=column)
        step = metrics["step"].to_numpy(dtype=float)
        alpha = metrics["alpha_local"].to_numpy(dtype=float)
        x = effective_step(step, spec["scale"]) if folded else step
        baseline = effective_step(BASELINE_STEP, spec["scale"]) if folded else BASELINE_STEP
        plot_series(ax, x, alpha, color=spec["color"], linewidth=1.8, label=spec["label"], baseline=baseline)
        if folded:
            folded_baseline(ax, BASELINE_STEP, spec["scale"], spec["color"])
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, label=r"$\alpha=1$")
    if folded:
        style_axis(ax)
    else:
        mark_baseline(ax)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"effective time $u=(lr/10^{-6})t$" if folded else "training step $t$")
    ax.set_ylabel(r"local exponent $\alpha(t)=-d\log d/d\log t$")
    ax.set_title(rf"M4 local exponent $\alpha(t)$: {metric_name}")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    axis_name = "effective-step" if folded else "step"
    output = figure_path("lr-ablation", "local-alpha", axis_name, "runs-m4", suffix)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_normalized_folded(column: str, metric_name: str, suffix: str) -> Path:
    """Render folded alpha with each run's step-30 fold baseline emphasized.

    ``alpha`` is a log-log slope, so normalizing ``d`` by a constant is a
    no-op; this variant differs from ``plot_metric(folded=True)`` only in the
    per-run dashed fold markers kept for the doc's step-30 narrative.
    """
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for run_id, spec in lr_runs():
        _, metrics = load_run_metrics(run_id, RUNS, column=column)
        step = metrics["step"].to_numpy(dtype=float)
        alpha = metrics["alpha_local"].to_numpy(dtype=float)
        x = effective_step(step, spec["scale"])
        baseline_x = effective_step(BASELINE_STEP, spec["scale"])
        plot_series(ax, x, alpha, color=spec["color"], linewidth=1.8, label=spec["label"], baseline=baseline_x)
        ax.axvline(baseline_x, color=spec["color"], ls="--", lw=0.8, alpha=0.7)
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, label=r"$\alpha=1$")
    style_axis(ax)
    ax.set_xlabel(r"effective time $u=(lr/10^{-6})t$")
    ax.set_ylabel(r"local exponent $\alpha(t)$")
    ax.set_title(rf"M4 $\alpha(t)$: {metric_name}; step-30 fold baselines")
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
