"""Shared figure naming and baseline plotting helpers."""

from pathlib import Path

import pandas as pd

from .paths import FIGURES_ROOT


def figure_path(
    family: str,
    y: str,
    x: str,
    scope: str,
    variant: str | None = None,
    root: Path = FIGURES_ROOT,
) -> Path:
    """Return a stable semantic figure path: ``y_vs_x__scope__variant.png``."""
    name = f"{y}_vs_{x}__{scope}"
    if variant:
        name += f"__{variant}"
    directory = root / family
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.png"


def split_baseline(
    frame: pd.DataFrame, baseline_step: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a plotted table into faded pre-baseline and primary segments."""
    return frame[frame["step"] < baseline_step], frame[frame["step"] >= baseline_step]
