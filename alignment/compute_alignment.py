#!/usr/bin/env python3
"""Compute the OPD alignment metrics d_t and v_t from pulled W&B history.

Inputs : alignment/runs/<run_id>__history.csv written by pull_wandb_metrics.py
Outputs: alignment/derived/<run_id>__alignment.csv (per-step series)
         alignment/derived/summary.json          (headline numbers)

Definitions (Fu et al. 2026, Eq. 6/7; see docs/2026-09-06-absorption-rate-spectral-theory.md):

    d_t = mean_i |log pi_T(y_i|s_i) - log pi_theta(y_i|s_i)|        (Eq. 6)
    v_t = (d_t - d_{t+1}) / d_t                                     (Eq. 7)

The trainer logs the signed per-token reward rm_scores = log pi_T - log pi_theta
(sampled-token run: full reverse-KL k1 estimator; top-16 run: top-k restricted
version). Its negative mean over visited tokens, -opd/token_reward_mean, is the
mean per-token reverse KL in nats -- a monotone distance proxy for d_t. Because
the proxy is a KL-type (squared) distance rather than an L1 mean, absolute
levels are not comparable across the two runs; the decay *shape* is what this
analysis reads.

v_t is estimated two ways:
  * v_direct : the raw one-step ratio (d_t - d_{t+1}) / d_t (noisy).
  * v_geo    : the paper's estimator -- slope of a linear fit of log d over the
               geometric window [t/1.4, 1.4t], v = 1 - exp(slope). Available
               only where the full window fits inside [t_start, t_end].
Both start at --t-start (default 30): gradient clipping is active before that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GEO_FACTOR = 1.4  # paper's geometric window [t/1.4, 1.4t]
DISTANCE_COLUMN = "opd/token_reward_mean"


def run_label(meta: dict) -> str:
    """Human label from the run config: top-k reward mode vs sampled token."""
    top_k = meta.get("config_flat", {}).get("distillation.fsdp_teacher.log_prob_top_k", 0)
    return f"top{int(top_k)}" if top_k else "sampled-token"


def geometric_absorption(steps: np.ndarray, dist: np.ndarray, t_start: int, factor: float) -> np.ndarray:
    """v_t = 1 - exp(slope of log d over the geometric window [t/factor, factor*t])."""
    v = np.full(steps.shape, np.nan)
    log_d = np.log(dist)
    for i, t in enumerate(steps):
        lo, hi = t / factor, t * factor
        window = (steps >= lo) & (steps <= hi)
        if window.sum() < 3:
            continue
        x, y = steps[window].astype(float), log_d[window]
        slope = np.polyfit(x, y, 1)[0]
        v[i] = 1.0 - math.exp(slope)
    return v


def power_law_exponent(steps: np.ndarray, dist: np.ndarray) -> float | None:
    """alpha in d ~ t^-alpha, least squares on (log t, log d) from t_start on."""
    mask = (steps >= 2) & np.isfinite(dist) & (dist > 0)
    if mask.sum() < 5:
        return None
    return float(-np.polyfit(np.log(steps[mask].astype(float)), np.log(dist[mask]), 1)[0])


def analyze_run(history_csv: Path, t_start: int, v_target_step: int) -> dict | None:
    df = pd.read_csv(history_csv)
    if DISTANCE_COLUMN not in df.columns:
        print(f"[skip] {history_csv.name}: no {DISTANCE_COLUMN} column")
        return None

    df = df.sort_values("_step").reset_index(drop=True)
    steps = df["_step"].to_numpy(dtype=float)
    dist = -df[DISTANCE_COLUMN].to_numpy(dtype=float)  # signed reward -> distance proxy (nats/token)
    if t_start not in steps:
        print(f"[skip] {history_csv.name}: step {t_start} missing")
        return None

    d_start = float(dist[steps == t_start][0])
    d_rel = dist / d_start

    v_direct = np.full(steps.shape, np.nan)
    v_direct[:-1] = (dist[:-1] - dist[1:]) / dist[:-1]
    v_geo = geometric_absorption(steps, dist, t_start, GEO_FACTOR)

    out = pd.DataFrame(
        {
            "step": steps.astype(int),
            "d_nats": dist,
            "d_rel": d_rel,
            "v_direct": v_direct,
            "v_geo": v_geo,
        }
    )

    tail = out.tail(5)
    d_end = float(tail["d_nats"].mean())

    def v_geo_at(step: int) -> float:
        return float(out.iloc[(out["step"] - step).abs().argmin()]["v_geo"])

    summary = {
        "run_id": history_csv.name.split("__")[0],
        "n_steps": int(out["step"].max()),
        "d_start_nats": d_start,
        "d_end_nats": d_end,
        "d_end_over_d_start": d_end / d_start,
        "v_geo_at_50": v_geo_at(50),
        "v_geo_at_200": v_geo_at(200),
        "v50_over_v200": v_geo_at(50) / v_geo_at(200),
        "power_law_alpha": power_law_exponent(steps[steps >= t_start], dist[steps >= t_start]),
    }
    return {"frame": out, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "derived")
    parser.add_argument(
        "--t-start",
        type=int,
        default=30,
        help="First step counted (clip active before this)",
    )
    parser.add_argument("--v-target-step", type=int, default=50, help="Step for the headline v readout")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    histories = sorted(args.runs_dir.glob("*__history.csv"))
    if not histories:
        print(
            f"error: no *__history.csv under {args.runs_dir} (run pull_wandb_metrics.py first)",
            file=sys.stderr,
        )
        return 2

    all_summaries = []
    labels = {}
    for csv_path in histories:
        result = analyze_run(csv_path, args.t_start, args.v_target_step)
        if result is None:
            continue
        run_id = result["summary"]["run_id"]
        out_csv = args.out_dir / f"{run_id}__alignment.csv"
        result["frame"].to_csv(out_csv, index=False)
        meta_path = csv_path.with_name(f"{run_id}__meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        labels[run_id] = run_label(meta)
        all_summaries.append({**result["summary"], "label": labels[run_id]})
        print(f"[compute] {run_id} -> {out_csv}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({"labels": labels, "runs": all_summaries}, indent=2))
    print(f"[compute] summary -> {summary_path}")

    header = f"{'run':<14} {'label':<14} {'d_end/d30':>10} {'v50':>8} {'v200':>8} {'v50/v200':>9} {'alpha':>7}"
    print(header)
    for s in all_summaries:
        print(
            f"{s['run_id']:<14} {s['label']:<14} {s['d_end_over_d_start']:>10.3f} "
            f"{s['v_geo_at_50']:>8.4f} {s['v_geo_at_200']:>8.4f} "
            f"{s['v50_over_v200']:>9.2f} {s['power_law_alpha']:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
