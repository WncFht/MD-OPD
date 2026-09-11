"""Load local experiment data into stable analysis inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

VALIDATION_TASKS = {
    "AIME25": "val-core/AIME25/acc/mean@16",
    "AMC23": "val-core/AMC23/acc/mean@16",
    "MATH-500": "val-core/MATH-500/acc/mean@16",
}


@dataclass(frozen=True)
class RunHistory:
    """One W&B history table together with its run metadata."""

    run_id: str
    frame: pd.DataFrame
    meta: dict


@cache
def load_run_history(run_id: str, runs_dir: Path) -> RunHistory:
    """Load ``<run_id>__history.csv`` and its optional metadata JSON.

    Cached per process: pipelines load the same history for several panels.
    Callers must not mutate ``history.frame`` in place.
    """
    csv_path = runs_dir / f"{run_id}__history.csv"
    meta_path = runs_dir / f"{run_id}__meta.json"
    frame = pd.read_csv(csv_path).sort_values("_step").reset_index(drop=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return RunHistory(run_id=run_id, frame=frame, meta=meta)


def load_run_histories(run_ids: list[str], runs_dir: Path) -> list[RunHistory]:
    """Load several local histories in the requested order."""
    return [load_run_history(run_id, runs_dir) for run_id in run_ids]
