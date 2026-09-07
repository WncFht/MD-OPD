"""Tail-form fitting metrics shared by local and paper curves."""

from __future__ import annotations

import numpy as np


def rmse_log(log_distance: np.ndarray, prediction: np.ndarray) -> float:
    """Return RMSE in log-distance space."""
    return float(np.sqrt(np.mean((np.log(prediction) - log_distance) ** 2)))


def _ols_2param(
    x: np.ndarray, log_distance: np.ndarray
) -> tuple[float, float, np.ndarray]:
    matrix = np.vstack([np.ones_like(x), x]).T
    coefficient, *_ = np.linalg.lstsq(matrix, log_distance, rcond=None)
    return coefficient[0], coefficient[1], coefficient[0] + coefficient[1] * x


def fit_power_law(t: np.ndarray, d: np.ndarray) -> dict:
    """Fit ``d = C t^-alpha``."""
    intercept, slope, prediction_log = _ols_2param(np.log(t), np.log(d))
    prediction = np.exp(prediction_log)
    return {
        "model": "power_law",
        "params": {"C": float(np.exp(intercept)), "alpha": float(-slope)},
        "rmse_log": rmse_log(np.log(d), prediction),
        "pred": prediction,
    }


def fit_stretched_exp(t: np.ndarray, d: np.ndarray) -> dict:
    """Fit ``d = C exp(-(t/tau)^gamma)`` on a gamma grid."""
    log_distance = np.log(d)
    best = None
    for gamma in np.geomspace(0.2, 2.0, 60):
        intercept, slope, prediction_log = _ols_2param(t**gamma, log_distance)
        if slope >= 0:
            continue
        prediction = np.exp(prediction_log)
        score = rmse_log(log_distance, prediction)
        if best is None or score < best["rmse_log"]:
            best = {
                "model": "stretched_exp",
                "params": {
                    "C": float(np.exp(intercept)),
                    "tau": float(np.exp(-np.log(-slope) / gamma)),
                    "gamma": float(gamma),
                },
                "rmse_log": score,
                "pred": prediction,
            }
    if best:
        gamma = best["params"]["gamma"]
        best["at_grid_boundary"] = bool(gamma <= 0.21 or gamma >= 1.9)
    return best


def fit_floor_exp(t: np.ndarray, d: np.ndarray) -> dict:
    """Fit ``d = d_inf + A exp(-t/tau)`` on a finite grid."""
    log_distance = np.log(d)
    best = None
    for tau in np.geomspace(5.0, 500.0, 60):
        for floor in np.geomspace(1e-4, 0.2 * d.min(), 40):
            residual = d - floor
            if np.any(residual <= 0):
                continue
            intercept, slope, prediction_log = _ols_2param(t, np.log(residual))
            if slope >= 0:
                continue
            prediction = floor + np.exp(prediction_log)
            score = rmse_log(log_distance, prediction)
            if best is None or score < best["rmse_log"]:
                best = {
                    "model": "floor_exp",
                    "params": {
                        "d_inf": float(floor),
                        "A": float(np.exp(intercept)),
                        "tau": float(tau),
                    },
                    "rmse_log": score,
                    "pred": prediction,
                }
    return best


def local_alpha(
    t: np.ndarray, d: np.ndarray, half_width: float = 0.12, min_points: int = 8
) -> np.ndarray:
    """Return the sliding local exponent ``-d log d / d log t``."""
    log_t, log_d = np.log10(t), np.log10(d)
    result = np.full(t.shape, np.nan)
    for index, value in enumerate(log_t):
        mask = np.abs(log_t - value) <= half_width
        if mask.sum() >= min_points:
            result[index] = -np.polyfit(log_t[mask], log_d[mask], 1)[0]
    return result
