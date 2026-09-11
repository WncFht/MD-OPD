"""Plot the MD-OPD topic runs: ppo_epochs ablation plus the mirror-descent variant."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from alignment.core.accuracy import accuracy_ylim, add_accuracy_baselines
from alignment.core.data import RunHistory, load_run_history
from alignment.core.metrics import direct_absorption, geometric_absorption
from alignment.core.plotting import (
    BASELINE_STEP,
    distance_definition,
    load_plot_style,
    mark_baseline,
    plot_series,
)
from alignment.core.tail import local_alpha

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "alignment" / "runs"
OUT_DIR = ROOT / "alignment" / "figures" / "md-opd"
SUMMARY_PATH = ROOT / "alignment" / "derived" / "md_opd_summary.json"
RUNS = {
    "5aub1k4h": ("one-shot", 1),
    "nh3jj9bt": ("4 mini steps", 4),
    "zyskh7f2": ("16 mini steps", 16),
    "6p579hnn": ("mirror descent (MDPO)", 4),
}


def plot_metric(
    ax: plt.Axes, frame: pd.DataFrame, column: str, label: str, color: str, marker: str | None = None
) -> None:
    """Plot one metric when it exists, preserving sparse validation points."""
    if column not in frame:
        return
    plot_series(
        ax,
        frame["_step"],
        frame[column],
        color=color,
        label=label,
        marker=marker,
        markersize=3.6,
        connect_missing=True,
    )


def apply_alignment_axis(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str, style, theme, baseline: bool = True
) -> None:
    """Apply shared axes, baseline, and early-training treatment."""
    if baseline:
        mark_baseline(ax)
    else:
        ax.set_xlim(left=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.apply_grid(ax, theme.grid)
    style.apply_frame(ax, theme.frame)


def alignment_figure(loaded: dict[str, RunHistory], colors: list[str], style, theme, folded: bool = False) -> Path:
    """Render distance, one-step absorption, and local power-law exponent."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for (run_id, (label, scale)), color in zip(RUNS.items(), colors, strict=True):
        frame = loaded[run_id].frame
        steps = frame["_step"].to_numpy(dtype=float)
        distance = -frame["opd/token_reward_mean"].to_numpy(dtype=float)
        x = steps * scale if folded else steps
        baseline = BASELINE_STEP * scale if folded else BASELINE_STEP
        plot_series(axes[0], x, distance, color=color, label=label, baseline=baseline)
        plot_series(axes[1], x, direct_absorption(distance), color=color, alpha=0.16, linewidth=0.8, baseline=baseline)
        plot_series(axes[1], x, geometric_absorption(steps, distance), color=color, baseline=baseline)
        plot_series(axes[2], x, local_alpha(steps, distance), color=color, label=label, baseline=baseline)

    xlabel = r"effective mini-step $u=k\,t$" if folded else "outer training step $t$"
    apply_alignment_axis(axes[0], r"Distance $d_t$", xlabel, "nats/token", style, theme, baseline=not folded)
    apply_alignment_axis(axes[1], r"Absorption $v_t$", xlabel, "relative drop", style, theme, baseline=not folded)
    apply_alignment_axis(
        axes[2], r"Local exponent $\alpha(t)$", xlabel, "dimensionless", style, theme, baseline=not folded
    )
    axes[0].set_yscale("log")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[2].set_yscale("symlog", linthresh=0.05)
    distance_definition(
        axes[0],
        extra=r"$v_t=(d_t-d_{t+1})/d_t$"
        "\n"
        r"$\alpha(t)=-\mathrm{d}\log d_t/\mathrm{d}\log t$",
    )
    axes[1].text(
        0.02,
        0.04,
        r"solid: geometric-window estimate"
        "\n"
        r"faint: raw one-step $v_t$",
        transform=axes[1].transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
    )
    if not folded:
        axes[0].text(
            BASELINE_STEP,
            0.96,
            "baseline",
            transform=axes[0].get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "MD-OPD runs: metrics on outer training steps"
        if not folded
        else "MD-OPD runs: metrics folded by actor mini-step count"
    )
    fig.tight_layout()
    output = OUT_DIR / ("dt-vt-alpha-folded.png" if folded else "dt-vt-alpha-outer-step.png")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    """Generate overview, validation, and systems-diagnostic figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style = load_plot_style()
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors(theme.name, len(RUNS))
    markers = style.series_markers(theme.name, len(RUNS))
    loaded = {run_id: load_run_history(run_id, RUNS_DIR) for run_id in RUNS}
    alignment_output = alignment_figure(loaded, colors, style, theme)
    folded_alignment_output = alignment_figure(loaded, colors, style, theme, folded=True)

    panels = [
        ("opd/token_reward_mean", "Signed token reward mean"),
        ("opd/adv_abs_mean", "Absolute advantage mean"),
        ("actor/entropy", "Policy entropy"),
        ("perf/throughput", "Reported throughput (tokens/s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for (run_id, (label, _)), color in zip(RUNS.items(), colors, strict=True):
        frame = loaded[run_id].frame
        for ax, (column, title) in zip(axes.flat, panels, strict=True):
            plot_metric(ax, frame, column, label, color)
            apply_alignment_axis(ax, title, "training step", "", style, theme)
    axes[0, 0].set_ylabel("nats/token")
    axes[0, 1].set_ylabel("nats/token")
    axes[1, 0].set_ylabel("nats/token")
    axes[1, 1].set_ylabel("tokens/s")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("MD-OPD runs: OPD and training diagnostics")
    fig.tight_layout()
    overview = OUT_DIR / "overview.png"
    fig.savefig(overview, dpi=220, bbox_inches="tight")
    plt.close(fig)

    tasks = ["AIME25", "AMC23", "MATH-500"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    accuracy_by_task = {task: [] for task in tasks}
    for (run_id, (label, _)), color, marker in zip(RUNS.items(), colors, markers, strict=True):
        frame = loaded[run_id].frame
        for ax, task in zip(axes, tasks, strict=True):
            column = f"val-core/{task}/acc/mean@16"
            if column not in frame:
                continue
            values = frame[column] * 100
            valid = values.notna()
            accuracy_by_task[task].append(values.loc[valid].to_numpy(dtype=float))
            plot_series(
                ax,
                frame.loc[valid, "_step"],
                values.loc[valid],
                color=color,
                label=label,
                linewidth=1.7,
                marker=marker,
                markersize=3.6,
                fade=False,
            )
    for ax, task in zip(axes, tasks, strict=True):
        mark_baseline(ax)
        ax.set_ylim(accuracy_ylim(accuracy_by_task[task], task))
        add_accuracy_baselines(ax, task=task, as_percent=True)
        ax.set_title(task)
        ax.set_xlabel("training step")
        ax.set_ylabel("accuracy mean@16 (%)")
        style.apply_grid(ax, theme.grid)
        style.apply_frame(ax, theme.frame)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("MD-OPD runs: validation by task", y=1.12)
    fig.tight_layout()
    validation_output = OUT_DIR / "validation.png"
    fig.savefig(validation_output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = []
    for run_id, (label, _) in RUNS.items():
        frame, meta = loaded[run_id].frame, loaded[run_id].meta
        row = {
            "run_id": run_id,
            "label": label,
            "state": meta["state"],
            "last_step": int(frame["_step"].max()),
            "rows": len(frame),
        }
        for column in ("opd/token_reward_mean", "opd/adv_abs_mean", "actor/entropy"):
            values = frame[column].dropna()
            row[f"{column}__first"] = float(values.iloc[0]) if len(values) else None
            row[f"{column}__last"] = float(values.iloc[-1]) if len(values) else None
        validation_columns = [f"val-core/{task}/acc/mean@16" for task in ("AIME25", "AMC23", "MATH-500")]
        validation_rows = frame[validation_columns].dropna(how="all")
        row["validation_last_mean16"] = float(validation_rows.iloc[-1].mean()) if len(validation_rows) else None
        row["validation_best_mean16"] = float(validation_rows.mean(axis=1).max()) if len(validation_rows) else None
        summary.append(row)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(alignment_output)
    print(folded_alignment_output)
    print(overview)
    print(validation_output)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
