#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alignment.config.runs import M4_RUNS
from alignment.core.accuracy import add_accuracy_baselines
from alignment.core.data import load_run_history
from alignment.core.metrics import cumulative_absorption, distance_proxy
from alignment.core.plotting import figure_path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment/runs"
OUT = figure_path("lr-ablation", "validation", "cumulative-absorption", "runs-m4", "eval-fit")


def main() -> Path:
    """Plot per-task validation accuracy against cumulative absorption -ln d_t."""
    runs = [(rid, spec["label"], spec["color"]) for rid, spec in M4_RUNS.items()]
    benches = [("AIME25", "AIME25"), ("AMC23", "AMC23"), ("MATH-500", "MATH-500")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axs = axs.ravel()
    for rid, label, color in runs:
        history = load_run_history(rid, RUNS)
        d = history.frame
        d = d[d["_step"] >= 30]
        x = cumulative_absorption(distance_proxy(history)[d.index.to_numpy()])
        for j, (short, name) in enumerate(benches):
            c = f"val-core/{short}/acc/mean@16"
            m = d[c].notna() if c in d else pd.Series(False, index=d.index)
            if m.sum() < 3:
                continue
            xx = x[m]
            yy = d.loc[m, c].to_numpy() * 100
            axs[j].plot(xx, yy, "o-", ms=3, lw=1.1, color=color, label=label)
            co = np.polyfit(xx, yy, 1)
            xs = np.linspace(xx.min(), xx.max(), 40)
            axs[j].plot(xs, np.polyval(co, xs), "--", lw=1, color=color, alpha=0.65)
            r2 = 1 - ((yy - np.polyval(co, xx)) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            axs[j].text(
                0.03,
                0.95 - (runs.index((rid, label, color)) * 0.1),
                f"{label}: $R^2={r2:.2f}$",
                transform=axs[j].transAxes,
                color=color,
                fontsize=8,
                va="top",
            )
    for j, (short, _) in enumerate(benches):
        add_accuracy_baselines(axs[j], task=short, as_percent=True)
    for j, (short, name) in enumerate(benches):
        axs[j].set_title(name)
        axs[j].set_xlabel(r"cumulative absorption $-\ln d_t$")
        axs[j].set_ylabel("accuracy (%)")
        axs[j].grid(axis="y", alpha=0.25)
    axs[3].axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Validation accuracy vs cumulative absorption across M4 runs", y=1.07)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main())
