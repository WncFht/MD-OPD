#!/usr/bin/env python3
"""Figure-19-style M4 comparison: validation, distance, and lr-collapsed distance."""

from pathlib import Path

import matplotlib.pyplot as plt
from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.metrics import compute_alignment_metrics, validation_mean

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment/runs"
OUT = ROOT / "alignment/figures/figure19_m4.png"
spec = [
    (rid, settings["label"], settings["scale"], settings["color"])
    for rid, settings in M4_RUNS.items()
    if rid != "tjjxnvw9"
]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for rid, label, scale, color in spec:
    history = load_run_history(rid, RUNS)
    d = history.frame
    metrics = compute_alignment_metrics(history)
    t = metrics["step"].to_numpy()
    dist = metrics["d_nats"].to_numpy()
    ax[1].plot(t, dist, color=color, lw=1.7, label=label)
    ax[2].plot(scale * t, dist, color=color, lw=1.7, label=label)
    valcols = [
        c for c in d.columns if c.startswith("val-core/") and c.endswith("/acc/mean@16")
    ]
    if valcols:
        vv = validation_mean(d)
        m = vv.notna()
        ax[0].plot(t[m], vv[m], color=color, marker="s", ms=3, lw=1.4, label=label)
ax[0].set_title("Average Validation Accuracy")
ax[0].set_xlabel("Training Step")
ax[0].set_ylabel("mean@16 (3-task average)")
ax[1].set_title(r"Distance $d_t$")
ax[1].set_yscale("log")
ax[1].set_xlabel("Training Step $t$")
ax[1].set_ylabel("nats/token")
ax[2].set_title(r"Absolute Distance $d_t$ (Collapsed)")
ax[2].set_yscale("log")
ax[2].set_xlabel(r"Effective Time $u=(lr/10^{-6})t$")
ax[2].set_ylabel(r"distance $d_t$ (nats/token)")
for a in ax:
    a.grid(True, axis="y", alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
if __name__ == "__main__":
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(OUT)
