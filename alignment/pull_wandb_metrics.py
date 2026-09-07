#!/usr/bin/env python3
"""Pull W&B run history to local CSVs for offline analysis.

Reusable across projects: point it at an entity/project and any number of run
ids; it writes one tidy CSV + one metadata JSON per run into the output dir.
Existing files are kept unless --force is passed, so re-runs are idempotent.

Usage:
    python alignment/pull_wandb_metrics.py \
        --entity wncfht --project One-Shot-OPD \
        --run-id mdxt04lm be00rj9e \
        --out alignment/runs

Credentials come from WANDB_API_KEY / ~/.netrc (same as the wandb CLI);
entity/project fall back to $WANDB_ENTITY / $WANDB_PROJECT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_KEYS = [
    # OPD alignment signal (per-token teacher-student log-prob gap statistics)
    "opd/token_reward_mean",
    "opd/adv_mean",
    "opd/adv_abs_mean",
    "opd/adv_rms",
    "opd/adv_std",
    "opd/adv_pos_mean",
    "opd/adv_neg_abs_mean",
    "opd/adv_pos_frac",
    "opd/teacher_logp_mean",
    "opd/teacher_nll_mean",
    "opd/student_nll_mean",
    "opd/student_entropy_mean",
    "opd/ce_minus_entropy_proxy",
    "opd/top_k_reward_mean",
    "opd/top_k_overlap_ratio",
    "opd/top_k_student_mass",
    "opd/top_k_teacher_mass",
    "opd/top_k_mode",
    # PPO/actor training health
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "critic/rewards/mean",
    "critic/score/mean",
    "actor/loss",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/grad_norm",
    "actor/lr",
    "actor/entropy",
    "actor/ppo_kl",
    # batch shape / speed
    "response_length/mean",
    "response/aborted_ratio",
    "perf/time_per_step",
    "perf/throughput",
    # validation checkpoints used by the Figure-19 and ln(d)-capability analyses
    "val-core/AIME25/acc/mean@16",
    "val-core/AMC23/acc/mean@16",
    "val-core/MATH-500/acc/mean@16",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (default $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="W&B project (default $WANDB_PROJECT)",
    )
    parser.add_argument(
        "--run-id", nargs="+", required=True, help="One or more W&B run ids"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Output directory",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Metric keys to pull (default: curated OPD list)",
    )
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="Text file with one metric key per line",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the CSV already exists",
    )
    return parser.parse_args()


def flatten_config(config: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, path))
        else:
            flat[path] = value
    return flat


def pull_run(
    api,
    entity: str,
    project: str,
    run_id: str,
    keys: list[str],
    out_dir: Path,
    force: bool,
) -> Path:
    import pandas as pd

    csv_path = out_dir / f"{run_id}__history.csv"
    meta_path = out_dir / f"{run_id}__meta.json"
    if csv_path.exists() and meta_path.exists() and not force:
        print(
            f"[skip] {run_id}: {csv_path} already exists (use --force to re-download)"
        )
        return csv_path

    run = api.run(f"{entity}/{project}/{run_id}")
    print(
        f"[pull] {run_id}: {run.name} (state={run.state}, last_step={run.lastHistoryStep})"
    )

    # scan_history with explicit keys only: runs with many metrics fail without it
    rows = list(run.scan_history(keys=keys, page_size=1000))
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all").sort_values("_step").reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(
        f"[pull] {run_id}: {df.shape[0]} steps x {df.shape[1] - 1} metrics -> {csv_path}"
    )

    meta = {
        "run_id": run_id,
        "name": run.name,
        "display_name": run.display_name,
        "state": run.state,
        "url": run.url,
        "created_at": run.created_at,
        "tags": list(run.tags),
        "notes": run.notes,
        "last_history_step": run.lastHistoryStep,
        "pulled_keys": keys,
        "config_flat": flatten_config(dict(run.config)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return csv_path


def main() -> int:
    args = parse_args()
    if not args.entity or not args.project:
        print(
            "error: --entity/--project (or $WANDB_ENTITY/$WANDB_PROJECT) are required",
            file=sys.stderr,
        )
        return 2

    keys = args.keys
    if args.keys_file:
        keys = [
            line.strip()
            for line in args.keys_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not keys:
        keys = DEFAULT_KEYS

    import wandb

    api = wandb.Api(timeout=120)
    args.out.mkdir(parents=True, exist_ok=True)
    for run_id in args.run_id:
        pull_run(api, args.entity, args.project, run_id, keys, args.out, args.force)

    print(f"[done] {len(args.run_id)} run(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
