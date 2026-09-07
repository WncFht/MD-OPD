"""Compatibility imports for scripts run from the alignment directory."""

from core.metrics import (
    alignment_summary,
    compute_alignment_metrics,
    distance_proxy,
    geometric_absorption,
    power_law_exponent,
)

__all__ = [
    "alignment_summary",
    "compute_alignment_metrics",
    "distance_proxy",
    "geometric_absorption",
    "power_law_exponent",
]
