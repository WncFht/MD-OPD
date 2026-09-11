"""Plot advantage and OPD reward diagnostics for the M4 runs."""

from pathlib import Path

import matplotlib.pyplot as plt

from alignment.config.runs import M4_RUNS
from alignment.core.data import load_run_history
from alignment.core.plotting import figure_path, mark_baseline, plot_series

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "alignment" / "runs"


def plot_advantage_diagnostics() -> Path:
    """Render signed, magnitude, and positive-fraction A diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for run_id, spec in M4_RUNS.items():
        frame = load_run_history(run_id, RUNS).frame
        if "opd/adv_abs_mean" not in frame:
            continue
        step = frame["_step"]
        color = spec["color"]
        label = spec["label"]
        signed = frame["opd/adv_mean"] if "opd/adv_mean" in frame else frame["opd/token_reward_mean"]
        plot_series(axes[0], step, signed, color=color, linewidth=1.5, label=label)
        plot_series(axes[1], step, frame["opd/adv_abs_mean"], color=color, linewidth=1.5, label=label)
        plot_series(axes[1], step, frame["opd/adv_rms"], color=color, linewidth=0.9, linestyle=":", alpha=0.8)
        plot_series(axes[2], step, frame["opd/adv_pos_frac"], color=color, linewidth=1.5, label=label)
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_title(r"signed mean $\mathbb{E}[A]$")
    axes[1].set_title(r"magnitude $\mathbb{E}|A|$ and RMS")
    axes[2].set_title(r"positive fraction $P(A>0)$")
    axes[0].set_ylabel("advantage / reward units")
    axes[2].set_ylabel("fraction")
    for axis in axes:
        mark_baseline(axis)
        axis.set_xlabel("training step $t$")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(r"M4 advantage diagnostics: signed gap, magnitude, and sign balance", y=1.02)
    fig.tight_layout()
    output = figure_path("lr-ablation", "advantage-diagnostics", "step", "runs-m4", "signed-magnitude-fraction")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(plot_advantage_diagnostics())
