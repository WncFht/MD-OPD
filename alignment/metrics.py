"""Reusable metric calculations for teacher-student alignment analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data import RunHistory

DISTANCE_COLUMN = "opd/token_reward_mean"
GEO_FACTOR = 1.4


def distance_proxy(history: RunHistory) -> np.ndarray:
    """Return the positive reverse-KL distance proxy from a W&B history."""
    if DISTANCE_COLUMN not in history.frame.columns:
        raise KeyError(f"history {history.run_id} has no {DISTANCE_COLUMN!r} column")
    return -history.frame[DISTANCE_COLUMN].to_numpy(dtype=float)


def geometric_absorption(steps: np.ndarray, distance: np.ndarray, factor: float = GEO_FACTOR) -> np.ndarray:
    """Estimate absorption from the log-distance slope in a geometric window."""
    values = np.full(steps.shape, np.nan)
    log_distance = np.log(distance)
    for index, step in enumerate(steps):
        window = (steps >= step / factor) & (steps <= step * factor)
        if window.sum() < 3:
            continue
        slope = np.polyfit(steps[window].astype(float), log_distance[window], 1)[0]
        values[index] = 1.0 - math.exp(slope)
    return values


def power_law_exponent(steps: np.ndarray, distance: np.ndarray) -> float | None:
    """Fit ``distance ~ step ** -alpha`` and return ``alpha``."""
    mask = (steps >= 2) & np.isfinite(distance) & (distance > 0)
    if mask.sum() < 5:
        return None
    return float(-np.polyfit(np.log(steps[mask]), np.log(distance[mask]), 1)[0])


def compute_alignment_metrics(history: RunHistory, t_start: int = 30) -> pd.DataFrame:
    """Build the canonical per-step alignment table for one run."""
    frame = history.frame
    if "_step" not in frame.columns:
        raise KeyError(f"history {history.run_id} has no '_step' column")
    steps = frame["_step"].to_numpy(dtype=float)
    distance = distance_proxy(history)
    if t_start not in steps:
        raise ValueError(f"history {history.run_id} has no baseline step {t_start}")
    baseline = float(distance[steps == t_start][0])
    relative = distance / baseline
    direct = np.full(steps.shape, np.nan)
    direct[:-1] = (distance[:-1] - distance[1:]) / distance[:-1]
    geometric = geometric_absorption(steps, distance)
    return pd.DataFrame(
        {
            "step": steps.astype(int),
            "d_nats": distance,
            "d_rel": relative,
            "v_direct": direct,
            "v_geo": geometric,
        }
    )


def alignment_summary(metrics: pd.DataFrame, run_id: str, t_start: int = 30) -> dict:
    """Summarize a canonical alignment table for reports and CLI output."""
    baseline = float(metrics.loc[metrics["step"] == t_start, "d_nats"].iloc[0])
    tail = metrics.tail(5)

    def value_at(step: int) -> float:
        row = metrics.iloc[(metrics["step"] - step).abs().argmin()]
        return float(row["v_geo"])

    v50 = value_at(50)
    v200 = value_at(200)
    return {
        "run_id": run_id,
        "n_steps": int(metrics["step"].max()),
        "d_start_nats": baseline,
        "d_end_nats": float(tail["d_nats"].mean()),
        "d_end_over_d_start": float(tail["d_nats"].mean() / baseline),
        "v_geo_at_50": v50,
        "v_geo_at_200": v200,
        "v50_over_v200": v50 / v200,
        "power_law_alpha": power_law_exponent(metrics["step"].to_numpy(float), metrics["d_nats"].to_numpy(float)),
    }
