#!/usr/bin/env python3
"""D1 tail-form discrimination (docs/2026-09-06-absorption-rate-spectral-theory.md §5 D1).

Fit three decay forms to the measured distance d_t (steps >= t_start) and
compare log-space residuals:

  power law        d = C t^-alpha            (spectral power law + amplitude decay)
  stretched exp    d = C exp(-(t/tau)^gamma) (wide near-uniform spectrum, glassy)
  floor + exp      d = d_inf + A exp(-t/tau) (single dominant mode + noise floor)

Each model reduces to ordinary least squares once its shape parameter(s) are
fixed (alpha: exact OLS in log-log; gamma/tau/d_inf: nested grid + OLS), so no
scipy is needed. Also emits a local exponent alpha(t) from a sliding log-log
regression, and renders tail_fits.png when --plot is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = REPO_ROOT / ".agents/skills/plot-skill/scripts/style_presets.py"


def load_plot_style():
    spec = importlib.util.spec_from_file_location("alignment_plot_style", STYLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot style: {STYLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rmse_log(log_d: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log(pred) - log_d) ** 2)))


def fit_power_law(t: np.ndarray, d: np.ndarray) -> dict:
    """ln d = ln C - alpha ln t, exact OLS."""
    A = np.vstack([np.ones_like(t), np.log(t)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(d), rcond=None)
    pred = np.exp(coef[0]) * t ** coef[1]  # coef[1] = -alpha
    return {
        "model": "power_law",
        "params": {"C": float(np.exp(coef[0])), "alpha": float(-coef[1])},
        "rmse_log": rmse_log(np.log(d), pred),
        "pred": pred,
    }


def _ols_2param(x: np.ndarray, log_d: np.ndarray) -> tuple[float, float, np.ndarray]:
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, log_d, rcond=None)
    return coef[0], coef[1], coef[0] + coef[1] * x


def fit_stretched_exp(t: np.ndarray, d: np.ndarray) -> dict:
    """ln d = ln C - (t/tau)^gamma; grid over gamma, exact OLS on (ln C, tau^-gamma).

    gamma < ~0.2 degenerates into "power law with drifting exponent" (t^gamma
    barely bends over the observed range), so the grid stops there; a boundary
    hit is flagged so the fit is not over-read.
    """
    log_d = np.log(d)
    g_lo, g_hi = 0.2, 2.0
    best = None
    for gamma in np.geomspace(g_lo, g_hi, 60):
        ln_c, slope, pred_log = _ols_2param(t**gamma, log_d)
        if slope >= 0:
            continue
        rmse = float(np.sqrt(np.mean((pred_log - log_d) ** 2)))
        if best is None or rmse < best["rmse_log"]:
            tau = float(np.exp(-np.log(-slope) / gamma))
            best = {
                "model": "stretched_exp",
                "params": {"C": float(np.exp(ln_c)), "tau": tau, "gamma": float(gamma)},
                "rmse_log": rmse,
                "pred": np.exp(pred_log),
            }
    if best is not None:
        gamma = best["params"]["gamma"]
        best["at_grid_boundary"] = bool(gamma <= g_lo * 1.05 or gamma >= g_hi * 0.95)
    return best


def fit_floor_exp(t: np.ndarray, d: np.ndarray) -> dict:
    """d = d_inf + A exp(-t/tau); grid over (tau, d_inf), exact OLS on ln(d - d_inf)."""
    log_d = np.log(d)
    best = None
    for tau in np.geomspace(5.0, 500.0, 60):
        for d_inf in np.geomspace(1e-4, 0.2 * d.min(), 40):
            y = d - d_inf
            if np.any(y <= 0):
                continue
            ln_a, slope, pred_log = _ols_2param(t, np.log(y))
            if slope >= 0:
                continue
            pred = d_inf + np.exp(pred_log)
            rmse = float(np.sqrt(np.mean((np.log(pred) - log_d) ** 2)))
            if best is None or rmse < best["rmse_log"]:
                best = {
                    "model": "floor_exp",
                    "params": {
                        "d_inf": float(d_inf),
                        "A": float(np.exp(ln_a)),
                        "tau": float(tau),
                    },
                    "rmse_log": rmse,
                    "pred": pred,
                }
    return best


def local_alpha(
    t: np.ndarray, d: np.ndarray, half_width: float = 0.12, min_points: int = 8
) -> np.ndarray:
    """Sliding local exponent: -d ln d / d ln t over a +-half_width window in log10 t."""
    log_t, log_d = np.log10(t), np.log10(d)
    out = np.full(t.shape, np.nan)
    for i, lt in enumerate(log_t):
        mask = np.abs(log_t - lt) <= half_width
        if mask.sum() < min_points:
            continue
        out[i] = -np.polyfit(log_t[mask], log_d[mask], 1)[0]
    return out


def analyze_run(csv_path: Path, t_start: int, plot: bool) -> dict:
    df = pd.read_csv(csv_path).sort_values("step")
    df = df[df["step"] >= t_start]
    t = df["step"].to_numpy(dtype=float)
    d = df["d_nats"].to_numpy(dtype=float)

    fits = [f(t, d) for f in (fit_power_law, fit_stretched_exp, fit_floor_exp)]
    fits.sort(key=lambda f: f["rmse_log"])
    alpha_local = local_alpha(t, d)

    run_id = csv_path.name.split("__")[0]
    pd.DataFrame({"step": t.astype(int), "local_alpha": alpha_local}).to_csv(
        csv_path.parent / f"{run_id}__local_alpha.csv", index=False
    )

    if plot:
        plot_fits(run_id, t, d, fits)

    return {
        "run_id": run_id,
        "n_points": int(t.size),
        "fits": [
            {k: f[k] for k in ("model", "params", "rmse_log") if k in f}
            | (
                {"at_grid_boundary": f["at_grid_boundary"]}
                if f.get("at_grid_boundary")
                else {}
            )
            for f in fits
        ],
        "best_model": fits[0]["model"],
        "local_alpha_at_50": float(alpha_local[np.argmin(np.abs(t - 50))]),
        "local_alpha_at_200": float(alpha_local[np.argmin(np.abs(t - 200))]),
    }


def plot_fits(run_id: str, t: np.ndarray, d: np.ndarray, fits: list[dict]) -> None:
    style = load_plot_style()
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", 4)
    out_dir = REPO_ROOT / "alignment" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(t, d, color=style.NEUTRALS["ink"], linewidth=1.4, label="measured $d_t$")
    labels = {
        "power_law": "power law $Ct^{-\\alpha}$",
        "stretched_exp": "stretched exp $Ce^{-(t/\\tau)^\\gamma}$",
        "floor_exp": "floor + exp $d_\\infty + Ae^{-t/\\tau}$",
    }
    for idx, fit in enumerate(fits):
        ax.plot(
            t,
            fit["pred"],
            color=colors[idx + 1],
            linewidth=1.6,
            linestyle="--",
            label=f"{labels[fit['model']]}  (rmse {fit['rmse_log']:.3f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Step (log)")
    ax.set_ylabel("Distance proxy (nats / token)")
    ax.set_title(f"Tail-form fits: {run_id}")
    style.apply_frame(ax, theme.frame)
    style.apply_grid(ax, theme.grid, axis="both")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = out_dir / f"tail_fits__{run_id}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fit] figure -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--derived-dir", type=Path, default=Path(__file__).resolve().parent / "derived"
    )
    parser.add_argument("--t-start", type=int, default=30)
    parser.add_argument(
        "--plot", action="store_true", help="Render tail_fits__<run>.png"
    )
    args = parser.parse_args()

    histories = sorted(args.derived_dir.glob("*__alignment.csv"))
    if not histories:
        print(
            f"error: no *__alignment.csv under {args.derived_dir} (run compute_alignment.py first)",
            file=sys.stderr,
        )
        return 2

    results = [analyze_run(p, args.t_start, args.plot) for p in histories]
    out_path = args.derived_dir / "tail_fits.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[fit] summary -> {out_path}\n")

    print(
        f"{'run':<14} {'best model':<14} {'rmse(power)':>12} {'rmse(stretch)':>14} {'rmse(floor)':>12} "
        f"{'alpha_loc50':>12} {'alpha_loc200':>13}"
    )
    for r in results:
        rmse = {f["model"]: f["rmse_log"] for f in r["fits"]}
        print(
            f"{r['run_id']:<14} {r['best_model']:<14} {rmse['power_law']:>12.4f} {rmse['stretched_exp']:>14.4f} "
            f"{rmse['floor_exp']:>12.4f} {r['local_alpha_at_50']:>12.3f} {r['local_alpha_at_200']:>13.3f}"
        )
    for r in results:
        for f in r["fits"]:
            if f["model"] == r["best_model"]:
                print(f"  best {r['run_id']}: {f['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
