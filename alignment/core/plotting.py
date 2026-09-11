"""Shared plotting helpers for alignment figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

FIGURES_ROOT = Path(__file__).resolve().parents[1] / "figures"
REPO_ROOT = FIGURES_ROOT.parents[1]
STYLE_PRESETS_PATH = REPO_ROOT / ".agents/skills/plot-skill/scripts/style_presets.py"

BASELINE_STEP = 30  # analysis baseline; also the gradient-clipping boundary
FADED_ALPHA = 0.25


def figure_path(
    family: str, metric: str, x_axis: str, scope: str, variant: str | None = None, root: Path = FIGURES_ROOT
) -> Path:
    """Return ``root/family/<metric>_vs_<x_axis>__<scope>[__<variant>].png``."""
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f"__{variant}" if variant else ""
    return directory / f"{metric}_vs_{x_axis}__{scope}{suffix}.png"


def load_plot_style():
    """Import the plot-skill ``style_presets`` module from outside the package."""
    spec = importlib.util.spec_from_file_location("alignment_plot_style", STYLE_PRESETS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot style: {STYLE_PRESETS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mark_baseline(axis, step: float = BASELINE_STEP, *, shaded: bool = True) -> None:
    """Shade the pre-baseline region and draw the baseline divider."""
    if shaded:
        axis.axvspan(0, step, color="#f3f3f3", zorder=0)
    axis.axvline(step, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_xlim(left=0)


def plot_series(
    axis,
    steps,
    values,
    *,
    color,
    label=None,
    baseline: float = BASELINE_STEP,
    fade: bool = True,
    connect_missing: bool = False,
    log_scale: bool = False,
    linestyle: str = "-",
    linewidth: float = 1.8,
    alpha: float = 1.0,
    marker=None,
    markersize: float = 3.0,
) -> None:
    """Plot a step series starting at step 0.

    With ``fade`` the segment left of ``baseline`` is drawn faint; accuracy
    series pass ``fade=False`` because their convention keeps early points and
    connecting lines fully visible (README: 不截断、不淡化).
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(steps)
    steps, values = steps[order], values[order]
    finite = np.isfinite(values)
    if connect_missing:
        first = np.argmax(finite) if finite.any() else len(finite)
        steps, values, finite = steps[first:], values[first:], finite[first:]
    style = {"color": color, "linestyle": linestyle, "marker": marker, "markersize": markersize, "linewidth": linewidth}
    segment = finite
    if fade:
        before = (steps < baseline) & finite
        axis.plot(steps[before], values[before], alpha=alpha * FADED_ALPHA, **style)
        segment = (steps >= baseline) & finite
    axis.plot(steps[segment], values[segment], alpha=alpha, label=label, **style)
    if log_scale:
        axis.set_yscale("log")


def distance_definition(axis, extra: str | None = None) -> None:
    """Annotate the canonical d_t definition in the lower-left corner."""
    text = (
        r"$A_t=\log\pi_T-\log\pi_\theta$"
        "\n"
        r"$d_t=-\mathrm{mean}(A_t)$"
        "\n"
        "d_t = -opd/token_reward_mean"
    )
    if extra:
        text += "\n" + extra
    axis.text(
        0.02,
        0.04,
        text,
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )
