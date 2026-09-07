"""Canonical locations for alignment inputs, results, and figures."""

from pathlib import Path

ALIGNMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ALIGNMENT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
NORMALIZED_ROOT = DATA_ROOT / "normalized"
RESULTS_ROOT = DATA_ROOT / "results"
FIGURES_ROOT = ALIGNMENT_ROOT / "figures"


def ensure_figure_dir(family: str) -> Path:
    """Create and return the output directory for one figure family."""
    path = FIGURES_ROOT / family
    path.mkdir(parents=True, exist_ok=True)
    return path
