"""Shared plotting helpers for alignment figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIGURES_ROOT = Path(__file__).resolve().parents[1] / "figures"

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
    """Import the plot-skill ``style_presets`` module (path set up in ``plots.core``)."""
    import style_presets

    return style_presets


def mark_baseline(axis, step: float = BASELINE_STEP, *, shaded: bool = True) -> None:
    """Shade the pre-baseline region and draw the baseline divider."""
    if shaded:
        axis.axvspan(0, step, color="#f3f3f3", zorder=0)
    axis.axvline(step, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_xlim(left=0)


def folded_baseline(axis, step: float, scale: float, color: str) -> None:
    """Draw one run's folded baseline divider ``u = scale * step``."""
    axis.axvline(step * scale, color=color, linestyle=":", linewidth=0.7, alpha=0.5)


def style_axis(axis) -> None:
    """Apply the shared axis finish: y-grid, no top/right spines, x from 0."""
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
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
