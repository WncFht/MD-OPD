#!/usr/bin/env python3
"""Compare the off-policy experiment W&B runs behind the 2026-09-09 analysis note."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plots.core.accuracy import accuracy_ylim, add_accuracy_baselines
from plots.core.data import load_run_history
from plots.core.metrics import (
    direct_absorption,
    geometric_absorption,
    normalize_at_step,
    validation_mean,
    value_at_step,
)
from plots.core.plotting import (
    BASELINE_STEP,
    distance_definition,
    figure_path,
    mark_baseline,
    plot_series,
)
from plots.core.tail import local_alpha

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "plots" / "runs"
SUMMARY_PATH = ROOT / "plots" / "derived" / "offpolicy_exp_summary.json"
# Set percentage bounds explicitly to zoom the accuracy panels, or leave None
# to derive a padded range from all valid accuracy observations.
ACCURACY_Y_LIMITS: tuple[float, float] | None = None
RUN_SPECS = {
    "5aub1k4h": {
        "label": "oneshot vanilla",
        "color": "#1f77b4",
    },
    "ceejzu26": {
        "label": "oneshot frozen no-ratio",
        "color": "#d95f02",
    },
    "mdxt04lm": {
        "label": "full vanilla",
        "color": "#2ca02c",
    },
    "0i38mr1k": {
        "label": "oneshot frozen vanilla",
        "color": "#9467bd",
    },
}


def style_axis(
    axis: plt.Axes,
    title: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Apply the shared step-zero and baseline presentation to one axis."""
    mark_baseline(axis)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def distance_validation_figure() -> Path:
    """Render distance, normalized distance, validation, and entropy panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    accuracy_values = []
    for run_id, spec in RUN_SPECS.items():
        frame = load_run_history(run_id, RUNS_DIR).frame
        steps = frame["_step"].to_numpy(dtype=float)
        distance = -frame["opd/token_reward_mean"].to_numpy(dtype=float)
        normalized = normalize_at_step(steps, distance, BASELINE_STEP)
        validation = validation_mean(frame).to_numpy(dtype=float) * 100
        accuracy_values.append(validation)

        plot_series(axes[0, 0], steps, distance, color=spec["color"], label=spec["label"], log_scale=True)
        plot_series(axes[0, 1], steps, normalized, color=spec["color"], label=spec["label"], log_scale=True)
        plot_series(
            axes[1, 0],
            steps,
            validation,
            color=spec["color"],
            label=spec["label"],
            marker="o",
            markersize=3.2,
            connect_missing=True,
            fade=False,
        )
        plot_series(
            axes[1, 1],
            steps,
            frame["actor/entropy"].to_numpy(dtype=float),
            color=spec["color"],
            label=spec["label"],
        )

    style_axis(
        axes[0, 0],
        r"Absolute distance $d_t$",
        "distance proxy (nats/token)",
    )
    style_axis(
        axes[0, 1],
        r"Normalized distance $d_t/d_{30}$",
        "dimensionless",
    )
    style_axis(
        axes[1, 0],
        "Validation accuracy",
        "3-task mean@16 (%)",
        y_limits=accuracy_ylim(accuracy_values, limits=ACCURACY_Y_LIMITS),
    )
    add_accuracy_baselines(axes[1, 0], as_percent=True)
    style_axis(axes[1, 1], "Policy entropy", "entropy (nats/token)")
    distance_definition(axes[0, 0])
    distance_definition(axes[0, 1])
    axes[1, 0].set_xlabel("training step $t$")
    axes[1, 1].set_xlabel("training step $t$")
    axes[0, 0].text(
        BASELINE_STEP,
        0.96,
        "baseline",
        transform=axes[0, 0].get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Requested W&B runs: distance and learning behavior", y=1.04)
    fig.tight_layout()
    output = figure_path("offpolicy-exp", "distance-validation", "step", "offpolicy-exp", "overview")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def validation_task_figure() -> Path:
    """Render each validation task instead of hiding it in a mean."""
    tasks = ["AIME25", "AMC23", "MATH-500"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(13, 4.2), sharey=False)
    accuracy_values_by_task = {task: [] for task in tasks}
    for run_id, spec in RUN_SPECS.items():
        frame = load_run_history(run_id, RUNS_DIR).frame
        steps = frame["_step"].to_numpy(dtype=float)
        for axis, task in zip(axes, tasks, strict=True):
            column = f"val-core/{task}/acc/mean@16"
            accuracy = frame[column].to_numpy(dtype=float) * 100
            accuracy_values_by_task[task].append(accuracy)
            plot_series(
                axis,
                steps,
                accuracy,
                color=spec["color"],
                label=spec["label"],
                marker="o",
                markersize=3.2,
                connect_missing=True,
                fade=False,
            )

    for axis, task in zip(axes, tasks, strict=True):
        style_axis(
            axis,
            task,
            "accuracy mean@16 (%)",
            y_limits=accuracy_ylim(accuracy_values_by_task[task], task, limits=ACCURACY_Y_LIMITS),
        )
        add_accuracy_baselines(axis, task=task, as_percent=True)
        axis.set_xlabel("training step $t$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Validation metrics by task", y=1.08)
    fig.tight_layout()
    output = figure_path("offpolicy-exp", "validation-tasks", "step", "offpolicy-exp", "mean16")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def absorption_alpha_figure() -> Path:
    """Render geometric and one-step absorption plus the local exponent of d_t."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=False)
    for run_id, spec in RUN_SPECS.items():
        frame = load_run_history(run_id, RUNS_DIR).frame
        steps = frame["_step"].to_numpy(dtype=float)
        distance = -frame["opd/token_reward_mean"].to_numpy(dtype=float)
        v_geo = geometric_absorption(steps, distance)
        v_direct = direct_absorption(distance)
        alpha = local_alpha(steps, distance)
        plot_series(
            axes[0],
            steps,
            v_direct,
            color=spec["color"],
            label="_nolegend_",
            linestyle=":",
            linewidth=0.9,
            alpha=0.55,
        )
        plot_series(axes[0], steps, v_geo, color=spec["color"], label=spec["label"])
        plot_series(axes[1], steps, alpha, color=spec["color"], label=spec["label"])

    style_axis(axes[0], r"Geometric absorption $v_t^{\rm geo}$", "relative drop per step")
    style_axis(axes[1], r"Local exponent $\alpha(t)$", "dimensionless")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_yscale("symlog", linthresh=0.05)
    axes[0].set_xlabel("training step $t$")
    axes[1].set_xlabel("training step $t$")
    fig.text(
        0.5,
        0.012,
        r"solid: geometric $v_t^{\rm geo}=1-\exp(\hat\beta_t)$; dotted: raw $v_t$; "
        r"$\hat\beta_t$ is the OLS slope of $\log d$ in $[t/1.4,\,1.4t]$"
        "\n"
        r"$\alpha(t)=-\mathrm{d}\log d_t/\mathrm{d}\log t$; "
        r"$A_t=\log\pi_T-\log\pi_\theta$; "
        r"$d_t=-\mathrm{mean}(A_t)$; `d_t = -opd/token_reward_mean`"
        "\n"
        r"non-positive $d_t$ samples omitted",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Absorption and local scaling of the signed-advantage distance proxy", y=1.08)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    output = figure_path("offpolicy-exp", "absorption-alpha", "step", "offpolicy-exp", "signed-adv-proxy")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def diagnostics_figure() -> Path:
    """Render OPD signal, overlap, gradient, and response diagnostics."""
    panels = [
        ("opd/top_k_overlap_ratio", "top-k overlap", False),
        ("actor/entropy", "policy entropy", False),
        ("actor/grad_norm", "actor gradient norm", True),
        ("response_length/mean", "response length", False),
        ("perf/throughput", "reported throughput", False),
        ("critic/score/mean", "critic score", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col")
    for run_id, spec in RUN_SPECS.items():
        frame = load_run_history(run_id, RUNS_DIR).frame
        steps = frame["_step"].to_numpy(dtype=float)
        for axis, (column, title, log_scale) in zip(axes.flat, panels, strict=True):
            plot_series(
                axis,
                steps,
                frame[column].to_numpy(dtype=float),
                color=spec["color"],
                label=spec["label"],
                log_scale=log_scale,
            )

    for axis, (_, title, _) in zip(axes.flat, panels, strict=True):
        ylabel = {
            "top-k overlap": "ratio",
            "policy entropy": "entropy (nats/token)",
            "actor gradient norm": r"$||g||_2$",
            "response length": "tokens",
            "reported throughput": "tokens/s",
            "critic score": "mean score",
        }[title]
        style_axis(axis, title, ylabel)
    for axis in axes[1]:
        axis.set_xlabel("training step $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Requested W&B runs: OPD and systems diagnostics", y=1.04)
    fig.tight_layout()
    output = figure_path("offpolicy-exp", "opd-diagnostics", "step", "offpolicy-exp", "overview")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output


def scalar_at(frame: pd.DataFrame, column: str, step: int) -> float | None:
    """Return a metric at an exact step, or null when that step was not logged."""
    if column not in frame:
        return None
    rows = frame.loc[frame["_step"] == step, column].dropna()
    return float(rows.iloc[0]) if len(rows) else None


def run_summary(run_id: str) -> dict:
    """Build reproducible headline statistics for one target run."""
    history = load_run_history(run_id, RUNS_DIR)
    frame, meta = history.frame, history.meta
    steps = frame["_step"].to_numpy(dtype=float)
    distance = -frame["opd/token_reward_mean"].to_numpy(dtype=float)
    baseline = value_at_step(steps, distance, BASELINE_STEP)
    v_geo = geometric_absorption(steps, distance)
    alpha = local_alpha(steps, distance)
    validation = validation_mean(frame).dropna()
    return {
        "run_id": run_id,
        "name": meta["name"],
        "state": meta["state"],
        "last_history_step": int(meta["last_history_step"]),
        "n_history_rows": int(len(frame)),
        "distance_d1": value_at_step(steps, distance, 1),
        "distance_d30": baseline,
        "distance_d100": value_at_step(steps, distance, 100),
        "distance_d_end": float(distance[-1]),
        "distance_end_over_d30": float(distance[-1] / baseline),
        "v_geo_at_50": value_at_step(steps, v_geo, 50),
        "alpha_at_50": value_at_step(steps, alpha, 50),
        "validation_first_mean16": float(validation.iloc[0]) if len(validation) else None,
        "validation_last_mean16": float(validation.iloc[-1]) if len(validation) else None,
        "validation_max_mean16": float(validation.max()) if len(validation) else None,
        "validation_max_step": (
            int(frame.loc[validation.index[validation.argmax()], "_step"]) if len(validation) else None
        ),
        "adv_abs_first": scalar_at(frame, "opd/adv_abs_mean", int(frame["_step"].iloc[0])),
        "adv_abs_last": scalar_at(frame, "opd/adv_abs_mean", int(frame["_step"].iloc[-1])),
        "top_k_overlap_first": float(frame["opd/top_k_overlap_ratio"].iloc[0]),
        "top_k_overlap_last": float(frame["opd/top_k_overlap_ratio"].iloc[-1]),
        "entropy_first": float(frame["actor/entropy"].iloc[0]),
        "entropy_last": float(frame["actor/entropy"].iloc[-1]),
    }


def write_summary() -> Path:
    """Write the target-run headline statistics as JSON."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_step": BASELINE_STEP,
        "runs": [run_summary(run_id) for run_id in RUN_SPECS],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return SUMMARY_PATH


def main() -> None:
    """Generate target-run figures and the machine-readable summary."""
    print(distance_validation_figure())
    print(validation_task_figure())
    print(absorption_alpha_figure())
    print(diagnostics_figure())
    print(write_summary())


if __name__ == "__main__":
    main()
