#!/usr/bin/env python3
"""Plot per-word reflection rates from word_stats.csv.

Uses the shared presets from .agents/skills/plot-skill/scripts/style_presets.py.
For each mode, draws a grouped bar chart: one panel per lexical category, words
on the x-axis, one colored series per model, y = rate per 1k words.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "plot-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from style_presets import TABLEAU_10, THEMES, apply_frame, apply_grid, apply_rc, palette  # noqa: E402

CATEGORY_ORDER = ["core", "branch", "verification", "transition"]


def series_colors(theme_name: str, n: int) -> list[str]:
    colors = palette(THEMES[theme_name].palette)
    if n <= len(colors):
        return colors[:n]
    extras = [color for color in TABLEAU_10.values() if color not in colors]
    return (colors + extras)[:n]


def load_rates(path: Path, include_checkpoints: bool) -> dict[tuple[str, str, str], dict[str, float]]:
    """Map (mode, category, word) -> {model: rate_per_1k_words}, skipping all-zero words."""
    rates: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if not include_checkpoints and re.search(r"global_step_\d+", model):
                continue
            rate = float(row["rate_per_1k_words"])
            if rate <= 0:
                continue
            rates[(row["mode"], row["category"], row["word"])][model] = rate
    return rates


def plot_mode(
    mode: str, rates: dict[tuple[str, str, str], dict[str, float]], current, theme_name: str, out_dir: Path
) -> Path | None:
    model_order = sorted({model for key in rates if key[0] == mode for model in rates[key]})
    if not model_order:
        return None
    colors = series_colors(theme_name, len(model_order))
    panels = []
    for category in CATEGORY_ORDER:
        entries = sorted(
            (key for key in rates if key[0] == mode and key[1] == category),
            key=lambda key: (-max(rates[key].values()), key[2]),
        )
        if entries:
            panels.append((category, entries))
    if not panels:
        return None

    max_words = max(len(entries) for _, entries in panels)
    fig_width = max(12.0, 0.82 * max_words + 3.0)
    fig_height = 2.9 * len(panels) + 1.3
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_width, fig_height))
    if len(panels) == 1:
        axes = [axes]

    for ax, (category, entries) in zip(axes, panels, strict=True):
        words = [key[2] for key in entries]
        x = np.arange(len(words))
        width = 0.8 / len(model_order)
        for idx, model in enumerate(model_order):
            values = [rates[key].get(model, 0.0) for key in entries]
            offset = (idx - (len(model_order) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[idx],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.95,
                label=model,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("rate / 1k words")
        ax.set_title(category, loc="left", fontsize=11, fontweight="bold")
        ax.set_axisbelow(True)
        ax.margins(x=0.01)
        apply_frame(ax, current.frame)
        apply_grid(ax, current.grid, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{mode}: reflection word rate per 1k words", fontsize=14, y=0.998)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(len(labels), 6),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"word_rates_{mode}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("reflection/results/word_stats.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reflection/figs"))
    parser.add_argument("--theme", default="tableau-paper", choices=sorted(THEMES))
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also plot model directories whose name contains global_step_N.",
    )
    args = parser.parse_args()

    current = apply_rc(args.theme)
    rates = load_rates(args.stats, args.include_checkpoints)
    modes = sorted({key[0] for key in rates})
    for mode in modes:
        out_path = plot_mode(mode, rates, current, args.theme, args.out_dir)
        if out_path:
            print(f"wrote: {out_path}")
        else:
            print(f"no data for mode: {mode}")


if __name__ == "__main__":
    main()
