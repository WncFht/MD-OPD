#!/usr/bin/env python3
"""M2: invert the spectral measure from measured d(t) curves (doc §2.9).

Model: d(t) = sum_k c_k exp(-rho_k * (t - t0)), t >= t0, rho_k = per-step decay
rate on a log-uniform grid. For the KL-proxy distance the underlying curvature
is lambda_k = rho_k / (2*eta) (2x rate, lambda-weighted amplitudes, doc §2.8);
for the paper's L1 metric lambda_k = rho_k / eta.

Fitting: non-negative least squares in *log space*, approximated by weighting
the linear residuals with 1/d_obs^2 (one IRLS step of the log-loss), solved
with multiplicative (Lee-Seung) updates. Tikhonov ridge on c via L-curve.
Validation: hold-out extrapolation onto the tail (fit [t0, t_split], predict
[t_split, end]) against a power-law baseline fitted on the same train range.

Outputs: alignment/derived/spectrum__<name>.csv, alignment/derived/inversion_summary.json,
         alignment/figures/spectrum__<name>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, HERE)

from fit_tail import fit_power_law  # noqa: E402

T0 = 30
ETA = 1e-6
RHO_MIN_FRAC = 1.0  # rho_min = RHO_MIN_FRAC / (T_end - T0)
RHO_MAX = 0.10
N_GRID = 64
N_ITER = 4000


def nnls_weighted(A: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float, iters: int = N_ITER) -> np.ndarray:
    """min_c || sqrt(w) (A c - y) ||^2 + ridge*||c||^2, c >= 0 (multiplicative updates)."""
    Aw = A * w[:, None]
    AtA = Aw.T @ A + ridge * np.eye(A.shape[1])
    Aty = Aw.T @ y
    c = np.full(A.shape[1], y.max() * 0.01)
    for _ in range(iters):
        c *= Aty / (AtA @ c + 1e-300)
        c = np.minimum(c, 10.0 * y.max())  # guard against overflow
    return c


def invert(name: str, df: pd.DataFrame, rate_factor: float, t_split: float) -> dict:
    d = df[(df["step"] >= T0)].sort_values("step")
    t = d["step"].to_numpy(dtype=float)
    y = d["value"].to_numpy(dtype=float)
    t_end = t.max()
    rho_min = RHO_MIN_FRAC / (t_end - T0)
    rho = np.geomspace(rho_min, RHO_MAX, N_GRID)
    A = np.exp(-np.outer(t - T0, rho))

    # L-curve over ridge strength
    w = 1.0 / np.maximum(y, 1e-12) ** 2
    lcurve = []
    for ridge in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        c = nnls_weighted(A, y, w, ridge * w.mean())
        res = float(np.sqrt(np.mean((np.log(np.maximum(A @ c, 1e-300)) - np.log(y)) ** 2)))
        lcurve.append({"ridge": ridge, "residual_log_rmse": res, "c_l2": float(np.linalg.norm(c))})
    # pick the smallest ridge whose residual is within 10% of the unregularized one
    res0 = lcurve[0]["residual_log_rmse"]
    chosen = next(
        (e for e in lcurve if e["residual_log_rmse"] <= 1.10 * max(res0, 1e-6)),
        lcurve[-1],
    )
    c = nnls_weighted(A, y, w, chosen["ridge"] * w.mean())

    # hold-out: refit on the train window only, extrapolate the tail
    train = t <= t_split
    A_tr, y_tr, w_tr = A[train], y[train], w[train]
    c_tr = nnls_weighted(A_tr, y_tr, w_tr, chosen["ridge"] * w_tr.mean())
    tail = ~train
    pred_tail = A[tail] @ c_tr
    rmse_tail = float(np.sqrt(np.mean((np.log(np.maximum(pred_tail, 1e-300)) - np.log(y[tail])) ** 2)))
    pl = fit_power_law(t[train], y[train])
    pred_pl = pl["params"]["C"] * t[tail] ** (-pl["params"]["alpha"])
    rmse_tail_pl = float(np.sqrt(np.mean((np.log(pred_pl) - np.log(y[tail])) ** 2)))

    # spectrum density on the log grid: mu'(rho) ~ c_k / (rho_k * dln rho)
    density = c / (rho * np.log(rho[1] / rho[0]))
    support = c > 0.02 * c.max()
    slope = None
    if support.sum() >= 5:
        slope = float(-np.polyfit(np.log(rho[support]), np.log(density[support]), 1)[0])

    spec = pd.DataFrame(
        {
            "rho_per_step": rho,
            "c_k": c,
            "density": density,
            "lambda_kl_reading": rho / rate_factor / ETA,
        }
    )
    spec.to_csv(HERE / "derived" / f"spectrum__{name}.csv", index=False)

    fig_out = plot(name, t, y, A @ c, t_split, rho, c, density, support)

    return {
        "name": name,
        "n_points": int(t.size),
        "ridge_chosen": chosen["ridge"],
        "residual_log_rmse": chosen["residual_log_rmse"],
        "lcurve": lcurve,
        "holdout": {
            "t_split": t_split,
            "nnls_rmse_log": rmse_tail,
            "power_law_rmse_log": rmse_tail_pl,
        },
        "density_loglog_slope": slope,
        "n_support_modes": int(support.sum()),
        "top_modes": [
            {
                "rho_per_step": float(r),
                "c_k": float(cc),
                "lambda_kl_reading": float(r / rate_factor / ETA),
            }
            for r, cc in sorted(zip(rho, c, strict=False), key=lambda kv: -kv[1])[:5]
            if cc > 0
        ],
        "figure": str(fig_out),
    }


def plot(name, t, y, recon, t_split, rho, c, density, support):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "alignment_plot_style",
        HERE.parent / ".agents/skills/plot-skill/scripts/style_presets.py",
    )
    style = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = style
    spec.loader.exec_module(style)
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", 3)
    styles = style.line_styles(theme.line_preset)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax1.plot(t, y, color=style.NEUTRALS["ink"], linewidth=1.4, label="measured $d_t$")
    ax1.plot(
        t,
        recon,
        color=colors[0],
        linewidth=1.6,
        linestyle="--",
        label="NNLS reconstruction",
    )
    ax1.axvline(
        t_split,
        color=style.NEUTRALS["baseline"],
        linewidth=1.0,
        linestyle=styles["reference"],
        alpha=0.7,
    )
    ax1.annotate(
        "hold-out",
        xy=(t_split, 0.85),
        xycoords=("data", "axes fraction"),
        color=style.NEUTRALS["baseline"],
        fontsize=8,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("distance (nats/token)")
    ax1.set_title("spectral reconstruction", fontsize=10)
    style.apply_frame(ax1, theme.frame)
    style.apply_grid(ax1, theme.grid, axis="y")
    ax1.legend(fontsize=8)

    ax2.plot(
        rho[support],
        density[support],
        "o",
        color=colors[0],
        markersize=4.5,
        label="inverted density",
    )
    ax2.plot(
        rho[~support],
        density[~support],
        "o",
        color=style.NEUTRALS["muted"],
        markersize=3,
        label="below support",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"per-step rate $\rho$ (= $2\eta\lambda$, KL reading)")
    ax2.set_ylabel(r"spectral density $\mu'(\rho)$")
    ax2.set_title("inverted spectrum", fontsize=10)
    style.apply_frame(ax2, theme.frame)
    style.apply_grid(ax2, theme.grid, axis="both")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = HERE / "figures" / f"spectrum__{name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    targets = []
    # our two runs (KL proxy: mode rate = 2*eta*lambda)
    for run_id, label in (
        ("be00rj9e", "ours_top16_kl"),
        ("mdxt04lm", "ours_sampled_token_kl"),
    ):
        df = pd.read_csv(HERE / "derived" / f"{run_id}__alignment.csv")
        targets.append((label, df[["step", "d_nats"]].rename(columns={"d_nats": "value"}), 2.0))
    # the paper's Full-Data L1 curve (digitized): rate = eta*lambda
    paper = load_paper_full_data()
    if paper is not None:
        targets.append(("paper_fulldata_l1", paper, 1.0))

    results = []
    for name, df, factor in targets:
        res = invert(name, df, factor, t_split=200.0)
        results.append(res)
        print(
            f"[invert] {name}: ridge={res['ridge_chosen']} res={res['residual_log_rmse']:.4f} "
            f"holdout NNLS={res['holdout']['nnls_rmse_log']:.4f} vs power-law "
            f"{res['holdout']['power_law_rmse_log']:.4f} slope={res['density_loglog_slope']}"
        )

    out = HERE / "derived" / "inversion_summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[invert] summary -> {out}")
    return 0


def load_paper_full_data() -> pd.DataFrame | None:
    dfs = list((HERE / "paper_curves").glob("5_digestion_rate_v2__panel0__00-3e-75*.csv"))
    if not dfs:
        return None
    df = pd.concat([pd.read_csv(p) for p in dfs]).sort_values("step")
    df["step"] = df["step"].round(2)
    return df.groupby("step", as_index=False)["value"].mean().reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
