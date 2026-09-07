#!/usr/bin/env python3
"""M1 on the paper's digitized curves: tail forms, alpha(t), and hump tests.

Inputs : alignment/paper_curves/*.csv (from extract_paper_figures.py)
Outputs: alignment/derived/paper_analysis.json
         alignment/figures/paper_tail_fits.png   (4 runs, log-log + fits)
         alignment/figures/alpha_comparison.png  (paper L1 alpha(t) vs ours, KL proxy)
         alignment/figures/hump_check.png        (H3a lr-fold / H3b frozen-state)

Series identity (curve colors are stable across the paper's figures):
  00-3e-75 dark blue = Full-Data (fig9) / On-Policy (5.3) / lr arm (see below)
  00-6c-77 teal      = 16-Shot / lr arm
  6b-4b-89 purple    = 4-Shot
  a6-18-2d red       = 1-Shot / Off-Policy / lr arm
  9a-67-00 olive     = third lr arm
The lr arms of 5A_lr_collapse are disambiguated by the empirical fold-matching
below (each collapsed curve is matched to a real-time curve with a fitted scale
k ~ lr ratio), which is self-validating.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, HERE)


from compute_alignment import geometric_absorption  # noqa: E402
from fit_tail import (  # noqa: E402
    fit_floor_exp,
    fit_power_law,
    fit_stretched_exp,
    local_alpha,
)

T_START = 30


def load(pattern: str) -> pd.DataFrame:
    dfs = list((HERE / "paper_curves").glob(pattern))
    if not dfs:
        raise SystemExit(f"missing extracted curves: {pattern}")
    df = pd.concat([pd.read_csv(p) for p in dfs]).sort_values("step")
    df["step"] = df["step"].round(2)
    return df.groupby("step", as_index=False)["value"].mean().reset_index(drop=True)


def load_concat(pattern: str) -> pd.DataFrame:
    """Concatenate complementary subpath segments (fig9 curves are split at the
    step-30 clipping line into a faded s0 and a main s1)."""
    dfs = list((HERE / "paper_curves").glob(pattern))
    if not dfs:
        raise SystemExit(f"missing extracted curves: {pattern}")
    df = pd.concat([pd.read_csv(p) for p in dfs]).sort_values("step")
    df["step"] = df["step"].round(2)
    return df.groupby("step", as_index=False)["value"].mean().reset_index(drop=True)


def load_main_branch(pattern: str) -> pd.DataFrame:
    """Load a curve that may span several subpath files; keep the branch with
    the largest dynamic range (clipped/ghost subpaths are nearly flat)."""
    dfs = list((HERE / "paper_curves").glob(pattern))
    if not dfs:
        raise SystemExit(f"missing extracted curves: {pattern}")
    best = None
    for p in dfs:
        df = pd.read_csv(p).sort_values("step")
        span = df["value"].max() / max(df["value"].min(), 1e-12)
        if best is None or span > best[0]:
            best = (span, df)
    return best[1].reset_index(drop=True)


def fit_all(steps: np.ndarray, dist: np.ndarray) -> list[dict]:
    fits = [f(steps, dist) for f in (fit_power_law, fit_stretched_exp, fit_floor_exp)]
    fits.sort(key=lambda f: f["rmse_log"])
    return [{k: f[k] for k in ("model", "params", "rmse_log") if k in f} for f in fits]


def interp_at(df: pd.DataFrame, step: float, col: str = "value") -> float:
    return float(np.interp(step, df["step"], df[col]))


def rate_curve(df: pd.DataFrame, t_start: int = T_START) -> pd.DataFrame:
    """Geometric-window absorption rate of a distance-like curve (factor 1.4)."""
    steps = df["step"].to_numpy(dtype=float)
    dist = df["value"].to_numpy(dtype=float)
    mask = steps >= t_start
    v = geometric_absorption(steps[mask], dist[mask], t_start, 1.4)
    return pd.DataFrame({"step": steps[mask], "v": v}).dropna().reset_index(drop=True)


def smooth_v(rc: pd.DataFrame, halflife: float = 3.0) -> pd.DataFrame:
    out = rc.copy()
    out["v_s"] = out["v"].ewm(halflife=halflife).mean()
    return out


def hump_step(rc: pd.DataFrame, lo: float = 40.0, hi: float = 120.0) -> dict:
    """Local maximum of the (smoothed) rate inside [lo, hi] vs the trough before it."""
    rc = smooth_v(rc)
    seg = rc[(rc["step"] >= lo) & (rc["step"] <= hi)]
    if seg.empty:
        return {}
    i = seg["v_s"].idxmax()
    pre = rc[(rc["step"] >= T_START) & (rc["step"] < seg["step"].min())]
    trough = float(pre["v_s"].min()) if not pre.empty else np.nan
    peak = float(seg.loc[i, "v_s"])
    rise = peak / trough if trough and trough > 0 else np.nan
    return {
        "hump_step": float(seg.loc[i, "step"]),
        "hump_v": peak,
        "pre_trough_v": trough,
        "hump_rise": rise,
    }


def fold_match(d_real: pd.DataFrame, d_fold: pd.DataFrame, k_range=None) -> dict:
    """Fit d_fold(u) ~= d_real(t = u / k): returns the best scale k and rmse."""
    if k_range is None:
        k_range = np.linspace(0.3, 2.6, 47)
    t1 = d_real["step"].to_numpy(dtype=float)
    v1 = d_real["value"].to_numpy(dtype=float)
    u2 = d_fold["step"].to_numpy(dtype=float)
    v2 = d_fold["value"].to_numpy(dtype=float)
    best: dict = {}
    for k in k_range:
        t = u2 / k
        m = (t >= t1.min()) & (t <= t1.max())
        if m.sum() < 30:
            continue
        pred = np.interp(t[m], t1, v1)
        rmse = float(np.sqrt(np.mean((np.log(pred) - np.log(v2[m])) ** 2)))
        if not best or rmse < best["rmse_log"]:
            best = {"k": round(float(k), 3), "rmse_log": rmse, "n": int(m.sum())}
    return best


def main() -> int:
    out_json = HERE / "derived" / "paper_analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"fig9": {}, "lr_collapse": {}, "online_offline": {}}

    # ---- Figure 9: distance (L1) + absorption rate, 4 runs ------------------
    fig9_colors = {
        "00-3e-75": "Full-Data (17k)",
        "00-6c-77": "16-Shot",
        "6b-4b-89": "4-Shot",
        "a6-18-2d": "1-Shot",
    }
    for hexc, label in fig9_colors.items():
        d = load_concat(f"5_digestion_rate_v2__panel0__{hexc}*.csv")
        d = d[d["step"] >= T_START]
        steps = d["step"].to_numpy(dtype=float)
        dist = d["value"].to_numpy(dtype=float)
        fits = fit_all(steps, dist)
        alpha_local = local_alpha(steps, dist)
        v = load_concat(f"5_digestion_rate_v2__panel1__{hexc}*.csv")
        v50, v200 = interp_at(v, 50.0), interp_at(v, 200.0)
        a50 = float(alpha_local[np.argmin(np.abs(steps - 50))])
        a200 = float(alpha_local[np.argmin(np.abs(steps - 200))])
        rc = pd.DataFrame({"step": v["step"], "v": v["value"] / 100.0})
        results["fig9"][label] = {
            "fits": fits,
            "local_alpha_50": a50,
            "local_alpha_200": a200,
            "v50_paper_pct": v50,
            "v200_paper_pct": v200,
            "v50_over_v200": v50 / v200,
            "identity_4a50a200": 4.0 * a50 / a200,
            "hump": hump_step(rc),
        }

    # ---- Appendix C.2: lr folding (H3a) --------------------------------------
    real_curves = {}
    for hexc in ("00-3e-75", "00-6c-77", "9a-67-00"):
        d = load_main_branch(f"5A_lr_collapse_v2__panel1__{hexc}*.csv")
        real_curves[hexc] = d[d["step"] >= T_START]
    fold_curves = {}
    for hexc in ("00-3e-75", "00-6c-77", "9a-67-00"):
        fold_curves[hexc] = load_main_branch(f"5A_lr_collapse_v2__panel2__{hexc}*.csv")

    match_table = {}
    for fhex, dfold in fold_curves.items():
        cands = []
        for rhex, dreal in real_curves.items():
            fm = fold_match(dreal, dfold)
            cands.append((fm["rmse_log"], rhex, fm))
        cands.sort()
        rmse, rhex, fm = cands[0]
        match_table[fhex] = {"matched_real": rhex, **fm}

    # arm identity from the fitted scale k (k = lr / 1e-6)
    k_by_hex = {fhex: m["k"] for fhex, m in match_table.items()}
    order = sorted(k_by_hex, key=lambda h: k_by_hex[h])
    arm_names = {}
    names = ["lr=0.5e-6", "lr=1e-6", "lr=2e-6"]
    for i, h in enumerate(order):
        arm_names[h] = names[min(i, 2)]

    lr_arms = {}
    for fhex, arm in arm_names.items():
        rhex = match_table[fhex]["matched_real"]
        d = real_curves[rhex]
        steps = d["step"].to_numpy(dtype=float)
        dist = d["value"].to_numpy(dtype=float)
        rc_t = rate_curve(d)
        rc_u = rate_curve(fold_curves[fhex])
        lr_arms[arm] = {
            "curve_hex": rhex,
            "fold_k": match_table[fhex]["k"],
            "fold_rmse_log": match_table[fhex]["rmse_log"],
            "hump_vs_t": hump_step(rc_t),
            "hump_u_from_t": hump_step(rc_t).get("hump_step", np.nan)
            * match_table[fhex]["k"],
            "hump_vs_u": hump_step(rc_u),
        }
    results["lr_collapse"] = dict(sorted(lr_arms.items()))

    # ---- §5.3: online vs off-policy (H3b) ------------------------------------
    for hexc, label in (("00-3e-75", "On-Policy"), ("a6-18-2d", "Off-Policy")):
        a = load_main_branch(f"5_online_offline__panel2__{hexc}*.csv")
        a["value"] = -a["value"]  # signed advantage -> distance-like
        rc = rate_curve(a)
        results["online_offline"][label] = {
            "n_points": int(len(a)),
            "hump": hump_step(rc),
            "v_at_50": interp_at(rc, 50.0, col="v"),
            "v_at_200": interp_at(rc, 200.0, col="v"),
        }

    out_json.write_text(json.dumps(results, indent=2))
    print(f"[analyze] summary -> {out_json}\n")

    hdr = f"{'run':<16} {'best fit':<13} {'a(50)':>7} {'a(200)':>7} {'v50/v200':>9}"
    hdr += f" {'4a50/a200':>10} {'hump@':>6} {'rise':>6}"
    print(hdr)
    for label, r in results["fig9"].items():
        h = r["hump"]
        print(
            f"{label:<16} {r['fits'][0]['model']:<13} {r['local_alpha_50']:>8.2f} {r['local_alpha_200']:>9.2f} "
            f"{r['v50_over_v200']:>9.2f} {r['identity_4a50a200']:>10.2f} {h.get('hump_step', float('nan')):>6.0f} "
            f"{h.get('hump_rise', float('nan')):>6.2f}"
        )
    print("\nlr folding (H3a): empirical fold-match + hump positions")
    for arm, r in results["lr_collapse"].items():
        ht, hu = r["hump_vs_t"], r["hump_vs_u"]
        print(
            f"  {arm:<10} k*={r['fold_k']:.2f} (rmse {r['fold_rmse_log']:.3f}) "
            f"hump@t={ht.get('hump_step', float('nan')):.0f} -> u={r['hump_u_from_t']:.0f} | "
            f"collapsed-curve hump@u={hu.get('hump_step', float('nan')):.0f}"
        )
    print("\nonline vs off-policy (H3b): rate-curve hump")
    for label, r in results["online_offline"].items():
        h = r["hump"]
        print(
            f"  {label:<11} v50={r['v_at_50']:.4f} v200={r['v_at_200']:.4f} "
            f"hump@{h.get('hump_step', float('nan')):.0f} rise={h.get('hump_rise', float('nan')):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
