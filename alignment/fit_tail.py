#!/usr/bin/env python3
"""Fit tail models for derived alignment tables."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from core.tail import fit_floor_exp, fit_power_law, fit_stretched_exp, local_alpha

ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = ROOT / ".agents/skills/plot-skill/scripts/style_presets.py"


def load_plot_style():
    """Load the repository plotting preset."""
    spec = importlib.util.spec_from_file_location("alignment_plot_style", STYLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plot style: {STYLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analyze_run(csv_path: Path, t_start: int, plot: bool) -> dict:
    """Fit all tail models for one derived alignment table."""
    frame = pd.read_csv(csv_path).sort_values("step")
    frame = frame[frame["step"] >= t_start]
    steps = frame["step"].to_numpy(float)
    distance = frame["d_nats"].to_numpy(float)
    fits = [
        fit_power_law(steps, distance),
        fit_stretched_exp(steps, distance),
        fit_floor_exp(steps, distance),
    ]
    fits.sort(key=lambda fit: fit["rmse_log"])
    alpha = local_alpha(steps, distance)
    run_id = csv_path.name.split("__")[0]
    pd.DataFrame({"step": steps.astype(int), "local_alpha": alpha}).to_csv(
        csv_path.parent / f"{run_id}__local_alpha.csv", index=False
    )
    if plot:
        plot_fits(run_id, steps, distance, fits)
    return {
        "run_id": run_id,
        "n_points": int(steps.size),
        "fits": [
            {k: fit[k] for k in ("model", "params", "rmse_log") if k in fit}
            | ({"at_grid_boundary": fit["at_grid_boundary"]} if fit.get("at_grid_boundary") else {})
            for fit in fits
        ],
        "best_model": fits[0]["model"],
        "local_alpha_at_50": float(alpha[(abs(steps - 50)).argmin()]),
        "local_alpha_at_200": float(alpha[(abs(steps - 200)).argmin()]),
    }


def plot_fits(run_id: str, steps, distance, fits: list[dict]) -> None:
    """Render measured distance and fitted tail models."""
    style = load_plot_style()
    theme = style.apply_rc("tableau-paper")
    colors = style.series_colors("tableau-paper", 4)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        steps,
        distance,
        color=style.NEUTRALS["ink"],
        linewidth=1.4,
        label="measured $d_t$",
    )
    names = {
        "power_law": "power law",
        "stretched_exp": "stretched exp",
        "floor_exp": "floor + exp",
    }
    for index, fit in enumerate(fits):
        ax.plot(
            steps,
            fit["pred"],
            color=colors[index + 1],
            linestyle="--",
            label=f"{names[fit['model']]} (rmse {fit['rmse_log']:.3f})",
        )
    ax.set(
        xlabel="Training step (log)",
        ylabel="Distance proxy (nats/token)",
        title=f"Tail-form fits: {run_id}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    style.apply_frame(ax, theme.frame)
    style.apply_grid(ax, theme.grid, axis="both")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = ROOT / "alignment/figures" / f"tail_fits__{run_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run tail fitting for all derived alignment tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, default=ROOT / "alignment/derived")
    parser.add_argument("--t-start", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    histories = sorted(args.derived_dir.glob("*__alignment.csv"))
    if not histories:
        raise SystemExit(f"no alignment tables under {args.derived_dir}")
    results = [analyze_run(path, args.t_start, args.plot) for path in histories]
    output = args.derived_dir / "tail_fits.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"[fit] summary -> {output}")
    for result in results:
        print(f"{result['run_id']:<14} {result['best_model']:<14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
