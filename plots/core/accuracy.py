"""Student/teacher accuracy baselines shared by alignment figures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import VALIDATION_TASKS

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
BASELINE_RUNS = {
    "student": "6uorrk2k",
    "teacher": "u87buult",
}
TASK_COLUMNS = VALIDATION_TASKS
TASK_ALIASES = {
    "AIME25": "AIME25",
    "AMC23": "AMC23",
    "MATH500": "MATH-500",
    "MATH-500": "MATH-500",
}
BASELINE_STYLES = {
    "student": {"color": "#4c566a", "linestyle": "--", "linewidth": 1.1},
    "teacher": {"color": "#a23e48", "linestyle": "--", "linewidth": 1.1},
}


def canonical_task(task: str) -> str:
    """Normalize task labels used by the different plotting pipelines."""
    key = task.strip().upper()
    try:
        return TASK_ALIASES[key]
    except KeyError as error:
        raise KeyError(f"No student/teacher baseline is available for task {task!r}") from error


@lru_cache(maxsize=8)
def _load_raw_baselines(runs_dir: str) -> dict[str, dict[str, float]]:
    root = Path(runs_dir)
    result: dict[str, dict[str, float]] = {}
    for role, run_id in BASELINE_RUNS.items():
        path = root / f"{run_id}__history.csv"
        frame = pd.read_csv(path)
        values = {}
        for task, column in TASK_COLUMNS.items():
            if column not in frame:
                raise KeyError(f"{path} has no {column!r}")
            observed = frame[column].dropna()
            if observed.empty:
                raise ValueError(f"{path} has no observed values for {column!r}")
            values[task] = float(observed.iloc[-1])
        result[role] = values
    return result


def accuracy_baseline_values(
    task: str | None = None,
    *,
    as_percent: bool = True,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> dict[str, float]:
    """Return student/teacher accuracy baselines for one task or the task mean."""
    raw = _load_raw_baselines(str(runs_dir))
    if task is None:
        values = {role: sum(task_values.values()) / len(task_values) for role, task_values in raw.items()}
    else:
        canonical = canonical_task(task)
        values = {role: task_values[canonical] for role, task_values in raw.items()}
    scale = 100.0 if as_percent else 1.0
    return {role: value * scale for role, value in values.items()}


def accuracy_ylim(
    values: Iterable[np.ndarray],
    task: str | None = None,
    *,
    limits: tuple[float, float] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[float, float]:
    """Return a percentage range covering observations and baselines.

    ``values`` must already be on the percent scale; student/teacher baselines
    are appended internally. ``limits=(lower, upper)`` selects a manual range
    that must cover every finite observation; otherwise the range is padded by
    ``max(2, 10% of the observed span)``.
    """
    arrays = [np.asarray(array, dtype=float) for array in values]
    arrays.append(np.array(list(accuracy_baseline_values(task, as_percent=True, runs_dir=runs_dir).values())))
    observed = np.concatenate([array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()])
    if observed.size == 0:
        return 0.0, 100.0
    if limits is not None:
        lower, upper = limits
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError(f"accuracy limits must satisfy 0 <= lower < upper <= 100, got {limits}")
        if observed.min() < lower or observed.max() > upper:
            raise ValueError(f"accuracy limits {limits} clip a valid observation")
        return float(lower), float(upper)
    lower, upper = float(observed.min()), float(observed.max())
    padding = max(2.0, (upper - lower) * 0.1)
    return max(0.0, lower - padding), min(100.0, upper + padding)


def add_accuracy_baselines(
    axis: plt.Axes,
    *,
    task: str | None = None,
    tasks: Iterable[str] | None = None,
    as_percent: bool = True,
    with_labels: bool = True,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Draw horizontal student/teacher baselines on an accuracy axis."""
    if task is not None and tasks is not None:
        raise ValueError("pass either task or tasks, not both")
    selected_tasks = list(dict.fromkeys(tasks or []))
    if task is not None:
        selected_tasks = [task]

    if not selected_tasks:
        values = accuracy_baseline_values(as_percent=as_percent, runs_dir=runs_dir)
        for role, value in values.items():
            label = f"{role} baseline" if with_labels else "_nolegend_"
            axis.axhline(value, label=label, zorder=2, **BASELINE_STYLES[role])
        return

    multiple_tasks = len(selected_tasks) > 1
    for selected_task in selected_tasks:
        canonical = canonical_task(selected_task)
        values = accuracy_baseline_values(
            canonical,
            as_percent=as_percent,
            runs_dir=runs_dir,
        )
        for role, value in values.items():
            suffix = f" ({canonical})" if multiple_tasks else ""
            label = f"{role} baseline{suffix}" if with_labels else "_nolegend_"
            axis.axhline(value, label=label, zorder=2, **BASELINE_STYLES[role])


def add_accuracy_baseline_segments(
    axis: plt.Axes,
    tasks: Iterable[str],
    centers: Iterable[float],
    half_width: float,
    *,
    as_percent: bool = True,
    with_labels: bool = True,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> None:
    """Draw task-specific baseline segments, useful for grouped bar charts."""
    labeled_roles: set[str] = set()
    for task, center in zip(tasks, centers, strict=True):
        try:
            values = accuracy_baseline_values(
                task,
                as_percent=as_percent,
                runs_dir=runs_dir,
            )
        except KeyError:
            continue
        for role, value in values.items():
            label = f"{role} baseline" if with_labels and role not in labeled_roles else "_nolegend_"
            axis.hlines(
                value,
                center - half_width,
                center + half_width,
                label=label,
                zorder=3,
                **BASELINE_STYLES[role],
            )
            labeled_roles.add(role)
