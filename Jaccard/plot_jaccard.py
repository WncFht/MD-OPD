#!/usr/bin/env python3
"""Plot D_J(L) curves from results/jaccard_diversity.csv using the plot-skill presets.

Figure layout:

- figures/all/ and figures/repetition-filtered/ hold the per-(mode, bench)
  figures and per-condition overviews.
- figures/repetition-filtered/steps_vs_1p7b-base_4b.png overlays the training
  checkpoints (``global_step_*``), Qwen3-1.7B-Base (no-thinking) and Qwen3-4B
  (thinking) in one figure per bench.

Model colors and markers are assigned globally so the same model keeps its
identity across every figure; the step checkpoints use a sequential Blues
gradient ordered by step number.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_plot_style():
    import importlib.util

    path = REPO_ROOT / ".agents/skills/plot-skill/scripts/style_presets.py"
    spec = importlib.util.spec_from_file_location("jaccard_plot_style", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot style: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


style = load_plot_style()

import matplotlib.pyplot as plt  # noqa: E402  (import after rc helper is available)

L_ORDER = ["256", "512", "1024", "2048", "4096", "8192", "16384", "full"]
L_TICKS = ["256", "512", "1024", "2048", "4096", "8192", "16384", "Full"]
CONDITIONS = {"all": "all", "no-repetition": "repetition-filtered"}
THEME = "tableau-paper"
STEP_PATTERN = re.compile(r"global_step_(\d+)")


def assign_styles(models: list[str]) -> dict[str, tuple[str, str]]:
    """Skill palette/markers, extended with unused Tableau tokens when n > 5."""
    colors = list(style.palette("tableau-paper"))
    for key in ("purple", "brown", "pink", "yellow"):
        if len(colors) >= len(models):
            break
        colors.append(style.TABLEAU_10[key])
    marks = list(style.markers("standard"))
    for mark in style.markers("paper-contrast"):
        if len(marks) >= len(models):
            break
        marks.append(mark)
    return {model: (colors[i], marks[i]) for i, model in enumerate(models)}


def load_results(path: Path) -> dict:
    """data[mode][bench][condition][model][L] = D_J."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if not row["D_J"]:
                continue
            data[row["mode"]][row["bench"]][row["condition"]][row["model"]][row["L"]] = float(row["D_J"])
    return data


def standalone_models(condition_data: dict) -> list[str]:
    return sorted(m for m in condition_data["all"] if not STEP_PATTERN.search(m))


def plot_curve(ax, values: dict[str, float], color: str, mark: str, dashed: bool = False) -> None:
    ys = [values.get(L) for L in L_ORDER]
    xs = [i for i, y in enumerate(ys) if y is not None]
    vs = [y for y in ys if y is not None]
    ax.plot(xs, vs, color=color, marker=mark, linewidth=1.8, markersize=5, linestyle="--" if dashed else "-")


def style_axes(ax, current) -> None:
    ax.set_xticks(range(len(L_ORDER)), L_TICKS, fontsize=8.5)
    ax.set_xlabel("Prefix length L (tokens)", fontsize=10)
    style.apply_frame(ax, current.frame)
    style.apply_grid(ax, current.grid, axis="y")


def plot_mode_bench(mode: str, bench: str, condition: str, condition_data: dict, styles: dict, out_dir: Path) -> Path:
    models = standalone_models(condition_data)
    current = style.apply_rc(THEME)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model in models:
        color, mark = styles[model]
        plot_curve(ax, condition_data[condition].get(model, {}), color, mark)
    handles = [plt.Line2D([], [], color=styles[m][0], marker="o", linewidth=1.8, label=m) for m in models]
    ax.set_ylabel("Jaccard distance $D_J$ (token 4-gram)")
    ax.set_title(f"{mode} · {bench}", fontsize=12)
    style_axes(ax, current)
    ax.legend(handles, [h.get_label() for h in handles], loc="lower right", fontsize=8.5)
    fig.tight_layout()
    out = out_dir / f"{mode}__{bench}__jaccard_diversity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overview(condition: str, data: dict, styles: dict, out_dir: Path) -> Path:
    benches = sorted({bench for mode in data for bench in data[mode]})
    modes = [m for m in ("no-thinking", "thinking") if m in data]
    fig, axes = plt.subplots(
        len(modes), len(benches), figsize=(4.9 * len(benches), 4.4 * len(modes)), sharey=True, squeeze=False
    )
    current = style.apply_rc(THEME)
    for row, mode in enumerate(modes):
        for col, bench in enumerate(benches):
            ax = axes[row][col]
            for model in standalone_models(data[mode][bench]):
                color, mark = styles[model]
                plot_curve(ax, data[mode][bench][condition].get(model, {}), color, mark)
            if col == 0:
                ax.set_ylabel(f"{mode}\n$D_J$", fontsize=10)
            if row == 0:
                ax.set_title(bench, fontsize=11)
            style_axes(ax, current)
    handles = [plt.Line2D([], [], color=styles[m][0], marker="o", linewidth=1.8, label=m) for m in sorted(styles)]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=min(len(handles), 6), fontsize=9)
    fig.suptitle(f"Token 4-gram Jaccard diversity $D_J(L)$ ({CONDITIONS[condition]})", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = out_dir / "overview__jaccard_diversity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_steps_vs_refs(data: dict, out_dir: Path) -> Path:
    """Training checkpoints vs reference models, one panel per bench, repetition-filtered."""
    import numpy as np

    condition = "no-repetition"
    benches = sorted(b for b in data["thinking"])
    step_models = sorted(
        (m for m in data["thinking"][benches[0]][condition] if STEP_PATTERN.search(m)),
        key=lambda m: int(STEP_PATTERN.search(m).group(1)),
    )
    step_colors = [plt.cm.Blues(v) for v in np.linspace(0.35, 0.95, len(step_models))]
    ref_specs = {
        "Qwen3-1.7B-Base": ("#2F2F2F", "s", True),  # ink, dashed; evaluated in no-thinking mode
        "Qwen3-4B": (style.TABLEAU_10["red"], "D", False),
    }
    current = style.apply_rc(THEME)

    fig, axes = plt.subplots(1, len(benches), figsize=(5.4 * len(benches), 4.8), sharey=True, squeeze=False)
    for col, bench in enumerate(benches):
        ax = axes[0][col]
        for model, color in zip(step_models, step_colors, strict=True):
            plot_curve(ax, data["thinking"][bench][condition].get(model, {}), color, "o")
        for model, (color, mark, dashed) in ref_specs.items():
            mode = "no-thinking" if model in data["no-thinking"] else "thinking"
            plot_curve(ax, data[mode][bench][condition].get(model, {}), color, mark, dashed)
        ax.set_title(bench, fontsize=12)
        style_axes(ax, current)
        if col == 0:
            ax.set_ylabel("Jaccard distance $D_J$ (token 4-gram)")
    handles = [
        plt.Line2D([], [], color=color, marker="o", linewidth=1.7, label=f"step {STEP_PATTERN.search(m).group(1)}")
        for m, color in zip(step_models, step_colors, strict=True)
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=ref_specs["Qwen3-1.7B-Base"][0],
            marker="s",
            linestyle="--",
            linewidth=1.7,
            label="Qwen3-1.7B-Base (no-thinking)",
        )
    )
    handles.append(
        plt.Line2D([], [], color=ref_specs["Qwen3-4B"][0], marker="D", linewidth=1.7, label="Qwen3-4B (thinking)")
    )
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=min(len(handles), 5), fontsize=9)
    fig.suptitle(f"Training checkpoints vs references, {CONDITIONS[condition]}", fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    out = out_dir / "steps_vs_1p7b-base_4b.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results" / "jaccard_diversity.csv"
    )
    parser.add_argument("--figures-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    data = load_results(args.results)

    all_models = sorted({m for mode in data for bench in data[mode] for m in standalone_models(data[mode][bench])})
    styles = assign_styles(all_models)

    written = []
    for condition, folder in CONDITIONS.items():
        cond_dir = args.figures_dir / folder
        cond_dir.mkdir(parents=True, exist_ok=True)
        for mode in data:
            for bench in sorted(data[mode]):
                written.append(plot_mode_bench(mode, bench, condition, data[mode][bench], styles, cond_dir))
        written.append(plot_overview(condition, data, styles, cond_dir))

    steps_dir = args.figures_dir / CONDITIONS["no-repetition"]
    steps_dir.mkdir(parents=True, exist_ok=True)
    written.append(plot_steps_vs_refs(data, out_dir=steps_dir))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
