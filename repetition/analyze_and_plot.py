#!/usr/bin/env python3
"""Analyze rollout repetition and generate the canonical figure set.

The detector follows the OPSA zlib idea, using overlapping 1024-character
windows so shorter and mid-response loops are observable.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import json
import lzma
import random
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WINDOW_SIZE = 1024
STRIDE = 256
THRESHOLD = 10.0
SAMPLE_PER_MODEL = 500
SEED = 42


def load_plot_style(repo_root: Path):
    import importlib.util

    path = repo_root / ".agents/skills/plot-skill/scripts/style_presets.py"
    spec = importlib.util.spec_from_file_location("repetition_plot_style", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot style: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ALGORITHMS = ("zlib", "gzip", "bz2", "lzma")


def compression_ratio(text: str, algorithm: str = "zlib") -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    if algorithm == "zlib":
        compressed = zlib.compress(raw, 9)
    elif algorithm == "gzip":
        compressed = gzip.compress(raw, compresslevel=9)
    elif algorithm == "bz2":
        compressed = bz2.compress(raw, compresslevel=9)
    elif algorithm == "lzma":
        compressed = lzma.compress(raw, preset=9)
    else:
        raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    return len(raw) / max(len(compressed), 1)


def window_ratios(text: str, algorithm: str = "zlib") -> tuple[np.ndarray, np.ndarray]:
    if len(text) <= WINDOW_SIZE:
        return np.array([0]), np.array([compression_ratio(text, algorithm)])
    starts = list(range(0, len(text) - WINDOW_SIZE + 1, STRIDE))
    last = len(text) - WINDOW_SIZE
    if starts[-1] != last:
        starts.append(last)
    ratios = [compression_ratio(text[start : start + WINDOW_SIZE], algorithm) for start in starts]
    return np.asarray(starts), np.asarray(ratios)


@dataclass
class Rollout:
    mode: str
    model: str
    bench: str
    source: Path
    example_id: str
    seed: str
    text: str
    starts: np.ndarray | None = None
    ratios: np.ndarray | None = None

    @property
    def label(self) -> str:
        return f"{self.mode}/{self.model}"

    def analyze(self, algorithm: str = "zlib") -> None:
        self.starts, self.ratios = window_ratios(self.text, algorithm)

    @property
    def repetition(self) -> bool:
        return bool(np.any(self.ratios >= THRESHOLD))

    @property
    def max_ratio(self) -> float:
        return float(np.max(self.ratios))

    @property
    def first_position(self) -> int | None:
        hits = np.flatnonzero(self.ratios >= THRESHOLD)
        return int(self.starts[hits[0]]) if len(hits) else None

    @property
    def first_position_frac(self) -> float | None:
        pos = self.first_position
        return pos / len(self.text) if pos is not None and self.text else None


def discover_roots(eval_root: Path) -> list[tuple[str, Path]]:
    roots = []
    for mode in ("no-thinking", "thinking"):
        mode_root = eval_root / mode
        for model_root in sorted(path for path in mode_root.iterdir() if path.is_dir()):
            if re.search(r"global_step_\d+", model_root.name):
                continue
            if list(model_root.glob("*.jsonl")):
                roots.append((mode, model_root))
    return roots


def load_rollouts(mode: str, model_root: Path) -> list[Rollout]:
    rollouts = []
    for path in sorted(model_root.glob("*.jsonl")):
        bench = path.name.split("_t", 1)[0]
        with path.open() as stream:
            for line_number, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                rollouts.append(
                    Rollout(
                        mode=mode,
                        model=model_root.name,
                        bench=bench,
                        source=path,
                        example_id=str(row.get("example_id", line_number)),
                        seed=str(row.get("seed", line_number)),
                        text=str(row.get("response", "")),
                    )
                )
    return rollouts


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", label)


def write_summary(all_models: dict[str, list[Rollout]], output_dir: Path) -> list[dict]:
    rows = []
    for label, rollouts in all_models.items():
        repeated = [item for item in rollouts if item.repetition]
        positions = [item.first_position for item in repeated]
        fractions = [item.first_position_frac for item in repeated]
        rows.append(
            {
                "model": label,
                "rollouts": len(rollouts),
                "repetition_count": len(repeated),
                "repetition_rate": len(repeated) / len(rollouts),
                "first_position_mean": float(np.mean(positions)) if positions else None,
                "first_position_median": float(np.median(positions)) if positions else None,
                "first_position_frac_mean": float(np.mean(fractions)) if fractions else None,
                "first_position_frac_median": float(np.median(fractions)) if fractions else None,
            }
        )
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_rate(summary: list[dict], figures: Path, style) -> None:
    labels = [row["model"] for row in summary]
    values = [100 * row["repetition_rate"] for row in summary]
    colors = ["#4E79A7" if label.startswith("no-thinking") else "#F28E2B" for label in labels]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Repetition rate (%)")
    ax.set_title("Rollout repetition rate across standalone models")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        "Blue: no-thinking; orange: thinking. Detector: window=1024, stride=256, threshold=10.",
        fontsize=8,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figures / "repetition_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compression_distribution(all_models: dict[str, list[Rollout]], figures: Path) -> None:
    bins = np.array([0, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, np.inf])
    labels = [
        "<2",
        "2-3",
        "3-4",
        "4-5",
        "5-6",
        "6-8",
        "8-10",
        "10-15",
        "15-20",
        "20-30",
        "30-50",
        "50-100",
        ">100",
    ]
    models = list(all_models)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.82 / max(len(models), 1)
    for i, label in enumerate(models):
        values = np.array([item.max_ratio for item in all_models[label]])
        counts, _ = np.histogram(values, bins=bins)
        pct = counts / max(len(values), 1) * 100
        ax.bar(x + (i - (len(models) - 1) / 2) * width, pct, width, label=label, alpha=0.85)
    ax.axvline(6.5, color="#6B7280", ls="--", lw=1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_xlabel("Maximum local compression ratio bin")
    ax.set_ylabel("Rollouts (%)")
    ax.set_title("Maximum local compression-ratio distribution")
    ax.grid(axis="y", ls="--", alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "compression_ratio_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_first_position_distribution(all_models: dict[str, list[Rollout]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, rollouts in all_models.items():
        flagged = [item for item in rollouts if item.repetition]
        pos = [item.first_position / 1000 for item in flagged if item.first_position is not None]
        frac = [item.first_position_frac for item in flagged if item.first_position_frac is not None]
        if pos:
            axes[0].hist(pos, bins=24, alpha=0.35, label=label)
        if frac:
            axes[1].hist(frac, bins=np.linspace(0, 1, 21), alpha=0.35, label=label)
    axes[0].set_xlabel("First repetition position (thousand characters)")
    axes[1].set_xlabel("First position / response length")
    for ax in axes:
        ax.set_ylabel("Flagged rollouts")
        ax.grid(axis="y", ls=":", alpha=0.7)
        ax.legend(fontsize=7)
    fig.suptitle("Distribution of repetition onset position")
    fig.tight_layout()
    fig.savefig(figures / "first_position_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_max_ratio_ecdf(all_models: dict[str, list[Rollout]], figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, rollouts in all_models.items():
        values = np.sort([item.max_ratio for item in rollouts])
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, lw=1.5, label=label)
    ax.axvline(10, color="#6B7280", ls="--", label="threshold=10")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Maximum local compression ratio (log scale)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("ECDF of maximum local compression ratio")
    ax.grid(True, ls=":", alpha=0.7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "max_compression_ratio_ecdf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_windows(all_models: dict[str, list[Rollout]], figures: Path) -> None:
    candidates = all_models.get("no-thinking/Qwen3-4B", next(iter(all_models.values())))
    positive = sorted([r for r in candidates if r.repetition], key=lambda r: r.max_ratio, reverse=True)[:5]
    negative = sorted([r for r in candidates if not r.repetition], key=lambda r: r.max_ratio)[:5]
    selected = [(r, "repetition") for r in positive] + [(r, "non-repetition") for r in negative]
    fig, axes = plt.subplots(5, 2, figsize=(13, 17), sharey=True)
    for ax, (item, label) in zip(axes.flat, selected, strict=True):
        x = item.starts / 1000
        ax.plot(
            x,
            item.ratios,
            color="#E15759" if label == "repetition" else "#4E79A7",
            lw=1.1,
        )
        ax.axhline(10, color="#6B7280", ls="--", lw=0.9)
        ax.set_xlabel("Window start (thousand chars)")
        ax.set_ylabel("Compression ratio")
        ax.set_title(
            f"{label} | {item.bench} | group={item.example_id}, rollout={item.seed} | max={item.max_ratio:.1f}",
            fontsize=9,
        )
        ax.grid(axis="y", ls=":", alpha=0.6)
    fig.suptitle("Five repetitive and five non-repetitive rollouts", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(figures / "five_repetition_examples.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def aggregate_curve(samples: list[Rollout], relative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if relative:
        max_position = max(float(item.starts[-1]) / max(len(item.text), 1) for item in samples)
        grid = np.linspace(0, max_position, 201)
    else:
        max_length = max(float(item.starts[-1]) for item in samples)
        grid = np.linspace(0, max_length, 240)
    matrix = np.full((len(samples), len(grid)), np.nan)
    for index, item in enumerate(samples):
        x = item.starts / max(len(item.text), 1) if relative else item.starts
        valid = grid <= x[-1]
        matrix[index, valid] = np.interp(grid[valid], x, item.ratios)
    return grid, np.nanmedian(matrix, axis=0), np.nanpercentile(matrix, 90, axis=0)


def plot_model(samples: list[Rollout], figures: Path, relative: bool) -> Path:
    label = samples[0].label
    fig, ax = plt.subplots(figsize=(10, 6))
    for item in samples:
        x = item.starts / max(len(item.text), 1) if relative else item.starts / 1000
        ax.plot(x, item.ratios, color="#4E79A7", alpha=0.035, linewidth=0.55)
    grid, median, p90 = aggregate_curve(samples, relative)
    if not relative:
        grid = grid / 1000
    ax.plot(grid, median, color="#E15759", linewidth=2.4, label="median")
    ax.plot(grid, p90, color="#F28E2B", linewidth=1.9, label="p90")
    ax.axhline(THRESHOLD, color="#6B7280", linestyle="--", linewidth=1.1, label="threshold=10")
    ax.set_yscale("log")
    ax.set_xlabel("Window start / response length" if relative else "Window start (thousand characters)")
    ax.set_ylabel("Window compression ratio (log scale)")
    axis_name = "relative position" if relative else "absolute position"
    ax.set_title(f"{label}: compression ratio across response length\n{len(samples)} sampled rollouts, {axis_name}")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend()
    fig.tight_layout()
    suffix = "relative" if relative else "absolute"
    path = figures / f"{slug(label)}__compression_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_sample_manifest(samples_by_model: dict[str, list[Rollout]], output_dir: Path) -> None:
    with (output_dir / "sample_manifest.csv").open("w", newline="") as stream:
        fields = [
            "model",
            "bench",
            "example_id",
            "seed",
            "response_chars",
            "repetition",
            "max_compression_ratio",
            "first_position",
            "first_position_frac",
            "source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, samples in samples_by_model.items():
            for item in samples:
                writer.writerow(
                    {
                        "model": label,
                        "bench": item.bench,
                        "example_id": item.example_id,
                        "seed": item.seed,
                        "response_chars": len(item.text),
                        "repetition": item.repetition,
                        "max_compression_ratio": item.max_ratio,
                        "first_position": item.first_position,
                        "first_position_frac": item.first_position_frac,
                        "source": item.source,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sample-per-model", type=int, default=SAMPLE_PER_MODEL)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    args = parser.parse_args()
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    style = load_plot_style(repo_root)
    style.apply_rc("tableau-paper")

    loaded_models: dict[str, list[Rollout]] = {}
    for mode, root in discover_roots(args.eval_root):
        rollouts = load_rollouts(mode, root)
        label = f"{mode}/{root.name}"
        loaded_models[label] = rollouts

    for algorithm in args.algorithms:
        all_models: dict[str, list[Rollout]] = {}
        samples_by_model: dict[str, list[Rollout]] = {}
        for label, rollouts in loaded_models.items():
            for item in rollouts:
                item.analyze(algorithm)
            all_models[label] = rollouts
            rng = random.Random(f"{SEED}:{algorithm}:{label}")
            samples_by_model[label] = rng.sample(rollouts, min(args.sample_per_model, len(rollouts)))

        algorithm_figures = figures / algorithm
        algorithm_figures.mkdir(parents=True, exist_ok=True)
        algorithm_output = args.output_dir / algorithm
        algorithm_output.mkdir(parents=True, exist_ok=True)
        summary = write_summary(all_models, algorithm_output)
        write_sample_manifest(samples_by_model, algorithm_output)
        plot_rate(summary, algorithm_figures, style)
        plot_compression_distribution(all_models, algorithm_figures)
        plot_max_ratio_ecdf(all_models, algorithm_figures)
        plot_first_position_distribution(all_models, algorithm_figures)
        plot_selected_windows(all_models, algorithm_figures)
        figure_index = {}
        for label, samples in samples_by_model.items():
            figure_index[label] = {
                "relative": plot_model(samples, algorithm_figures, relative=True).name,
                "absolute": plot_model(samples, algorithm_figures, relative=False).name,
            }
        (algorithm_figures / "figure_index.json").write_text(json.dumps(figure_index, indent=2))
        rollout_count = sum(len(v) for v in all_models.values())
        print(f"[{algorithm}] analyzed {rollout_count} rollouts across {len(all_models)} model configurations")


if __name__ == "__main__":
    main()
