#!/usr/bin/env python3
"""Plot the OPD alignment curves d_t and v_t with the shared plot-skill style.

Inputs : alignment/derived/<run_id>__alignment.csv from compute_alignment.py
Outputs: alignment/figures/alignment/*.png (paper Figure-9 layout, raw
         distance proxy, direct-vs-geometric absorption estimators)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from alignment.core.plotting import (
    BASELINE_STEP,
    figure_path,
    load_plot_style,
    mark_baseline,
    plot_series,
)

ALIGNMENT_ROOT = Path(__file__).resolve().parents[2]


def load_runs(derived_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for csv_path in sorted(derived_dir.glob("*__alignment.csv")):
        run_id = csv_path.name.split("__")[0]
        frames[run_id] = pd.read_csv(csv_path).sort_values("step")
    if not frames:
        raise SystemExit(f"error: no *__alignment.csv under {derived_dir} (run compute_alignment.py first)")
    return frames


def annotate_clipping(ax, style) -> None:
    mark_baseline(ax)
    ax.annotate(
        "clipping",
        xy=(BASELINE_STEP, 0.03),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color=style.NEUTRALS["baseline"],
        fontsize=8,
    )


def plot_fig9(frames: dict[str, pd.DataFrame], labels: dict[str, str], figures_dir: Path, style) -> None:
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", len(frames))
    styles = style.line_styles(theme.line_preset)

    fig, (ax_d, ax_v) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for idx, (run_id, df) in enumerate(frames.items()):
        plot_series(
            ax_d,
            df["step"],
            df["d_rel"],
            color=colors[idx],
            linewidth=1.8,
            linestyle=styles["primary"],
            label=labels[run_id],
        )
        plot_series(
            ax_v,
            df["step"],
            df["v_geo"] * 100,
            color=colors[idx],
            linewidth=1.8,
            linestyle=styles["primary"],
            label=labels[run_id],
        )

    ax_d.set_yscale("log")
    ax_d.set_xlabel("Training step $t$ (dimensionless)")
    ax_d.set_ylabel(r"normalised distance $d_t/d_{30}$ (dimensionless)")
    ax_d.set_title("Distance Left $d_t/d_{30}$")
    annotate_clipping(ax_d, style)

    ax_v.set_yscale("log")
    ax_v.set_xlabel("Training step $t$ (dimensionless)")
    ax_v.set_ylabel(r"absorption rate $v_t$ (% per step)")
    ax_v.set_title(r"Absorption Rate $v_t$ (%)")
    annotate_clipping(ax_v, style)

    for ax in (ax_d, ax_v):
        style.apply_frame(ax, theme.frame)
        style.apply_grid(ax, theme.grid, axis="y")
    handles, names = ax_d.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(frames), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    out = figure_path("alignment", "distance", "step", "runs-alignment", "norm-d30", figures_dir)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_distance_raw(frames: dict[str, pd.DataFrame], labels: dict[str, str], figures_dir: Path, style) -> None:
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", len(frames))
    styles = style.line_styles(theme.line_preset)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for idx, (run_id, df) in enumerate(frames.items()):
        plot_series(
            ax,
            df["step"],
            df["d_nats"],
            color=colors[idx],
            linewidth=1.8,
            linestyle=styles["primary"],
            label=f"{labels[run_id]}  $-\\overline{{A}}_t$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training step $t$ (dimensionless)")
    ax.set_ylabel("distance proxy $d_t$ (nats/token)")
    ax.set_title("Teacher-Student Distance $d_t$ (mean per-token reverse KL)")
    annotate_clipping(ax, style)
    style.apply_frame(ax, theme.frame)
    style.apply_grid(ax, theme.grid, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = figure_path("alignment", "distance", "step", "runs-alignment", "absolute", figures_dir)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_absorption_estimators(
    frames: dict[str, pd.DataFrame], labels: dict[str, str], figures_dir: Path, style
) -> None:
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", len(frames))
    styles = style.line_styles(theme.line_preset)

    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 3.8), squeeze=False)
    for idx, (run_id, df_all) in enumerate(frames.items()):
        ax = axes[0][idx]
        df = df_all
        ax.plot(
            df["step"],
            df["v_direct"] * 100,
            color=colors[idx],
            linewidth=0.9,
            alpha=0.35,
            linestyle=styles["primary"],
            label="one-step (raw)",
        )
        ax.plot(
            df["step"],
            df["v_geo"] * 100,
            color=colors[idx],
            linewidth=2.0,
            linestyle=styles["primary"],
            label=r"geometric window $[t/1.4,\,1.4t]$",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Training step $t$ (dimensionless)")
        if idx == 0:
            ax.set_ylabel(r"absorption rate $v_t$ (% per step)")
        ax.set_title(labels[run_id])
        annotate_clipping(ax, style)
        style.apply_frame(ax, theme.frame)
        style.apply_grid(ax, theme.grid, axis="y")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(r"Absorption Rate $v_t$: one-step ratio vs geometric-window estimate", y=1.0)
    fig.tight_layout()
    out = figure_path(
        "alignment",
        "absorption",
        "step",
        "runs-alignment",
        "estimators-direct-geo",
        figures_dir,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derived-dir", type=Path, default=ALIGNMENT_ROOT / "derived")
    parser.add_argument("--figures-dir", type=Path, default=ALIGNMENT_ROOT / "figures")
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=ALIGNMENT_ROOT / "derived" / "summary.json",
    )
    args = parser.parse_args()

    style = load_plot_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    frames = load_runs(args.derived_dir)

    labels = {run_id: run_id for run_id in frames}
    if args.labels_file.exists():
        summary = json.loads(args.labels_file.read_text())
        labels.update({rid: name for rid, name in summary.get("labels", {}).items()})

    plot_fig9(frames, labels, args.figures_dir, style)
    plot_distance_raw(frames, labels, args.figures_dir, style)
    plot_absorption_estimators(frames, labels, args.figures_dir, style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
