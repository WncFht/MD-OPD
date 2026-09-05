#!/usr/bin/env python3
"""Plot per-response reflection statistics from per_rollout.csv.

per_rollout.csv already contains only rollouts that passed the repetition
detector, so repeats are excluded by construction. For each mode, draws a 2x2
grid of bar charts, one bar per model:
  - mean reflection words per response
  - median reflection words per response
  - mean per-response rate per 1k words
  - share of responses containing at least one reflection word

Uses the shared presets from .agents/skills/plot-skill/scripts/style_presets.py;
model colors match the per-word chart (word_rates_*.png).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "plot-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from style_presets import TABLEAU_10, THEMES, apply_frame, apply_grid, apply_rc, palette  # noqa: E402


def series_colors(theme_name: str, n: int) -> list[str]:
    colors = palette(THEMES[theme_name].palette)
    if n <= len(colors):
        return colors[:n]
    extras = [color for color in TABLEAU_10.values() if color not in colors]
    return (colors + extras)[:n]


def load_responses(path: Path, include_checkpoints: bool) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """Map mode -> model -> (per-response counts, per-response rates per 1k words)."""
    data: dict[str, dict[str, tuple[list[float], list[float]]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if not include_checkpoints and re.search(r"global_step_\d+", model):
                continue
            counts, rates = data[row["mode"]][model]
            counts.append(float(row["lexical_count"]))
            rates.append(float(row["lexical_rate_per_1k_words"]))
    return data


METRICS = [
    ("mean reflection words / response", "words / response", lambda counts, rates: mean(counts), "{:.1f}"),
    ("median reflection words / response", "words / response", lambda counts, rates: median(counts), "{:.1f}"),
    ("mean rate / 1k words", "rate / 1k words", lambda counts, rates: mean(rates), "{:.1f}"),
    (
        "responses with ≥1 reflection word",
        "% of responses",
        lambda counts, rates: 100.0 * mean(count >= 1 for count in counts),
        "{:.1f}",
    ),
]


def plot_mode(
    mode: str, models: dict[str, tuple[list[float], list[float]]], current, theme_name: str, out_dir: Path
) -> Path:
    model_order = sorted(models)
    colors = series_colors(theme_name, len(model_order))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    for ax, (title, ylabel, stat, label_format) in zip(axes.flat, METRICS, strict=True):
        values = [stat(*models[model]) for model in model_order]
        bars = ax.bar(model_order, values, width=0.62, color=colors, edgecolor="white", linewidth=0.6, alpha=0.95)
        ax.bar_label(
            bars, labels=[label_format.format(value) for value in values], padding=2, fontsize=9, color="#2F2F2F"
        )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(values) * 1.15)
        ax.tick_params(axis="x", labelrotation=30, labelsize=9)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.set_axisbelow(True)
        apply_frame(ax, current.frame)
        apply_grid(ax, current.grid, axis="y")

    fig.suptitle(f"{mode}: per-response reflection statistics (repetition excluded)", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_response_{mode}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-rollout", type=Path, default=Path("reflection/results/per_rollout.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reflection/figs"))
    parser.add_argument("--theme", default="tableau-paper", choices=sorted(THEMES))
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also plot model directories whose name contains global_step_N.",
    )
    args = parser.parse_args()

    current = apply_rc(args.theme)
    data = load_responses(args.per_rollout, args.include_checkpoints)
    for mode in sorted(data):
        out_path = plot_mode(mode, data[mode], current, args.theme, args.out_dir)
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
