#!/usr/bin/env python3
"""M1/M2 figures: paper-curve tail fits, alpha(t) comparison, hump check, spectra."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from alignment.core.plotting import figure_path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parent
sys.path.insert(0, HERE)

from alignment.core.metrics import geometric_absorption  # noqa: E402
from alignment.core.tail import (  # noqa: E402
    fit_floor_exp,
    fit_power_law,
    fit_stretched_exp,
    local_alpha,
)

STYLE_PATH = ROOT / ".agents/skills/plot-skill/scripts/style_presets.py"
T_START = 30


def load_style():
    spec = importlib.util.spec_from_file_location("alignment_plot_style", STYLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_concat(name: str) -> pd.DataFrame:
    dfs = list((HERE / "paper_curves").glob(name))
    df = pd.concat([pd.read_csv(p) for p in dfs]).sort_values("step")
    df["step"] = df["step"].round(2)
    return df.groupby("step", as_index=False)["value"].mean().reset_index(drop=True)


def load_main_branch(name: str) -> pd.DataFrame:
    best = None
    for p in (HERE / "paper_curves").glob(name):
        df = pd.read_csv(p).sort_values("step")
        span = df["value"].max() / max(df["value"].min(), 1e-12)
        if best is None or span > best[0]:
            best = (span, df)
    return best[1].reset_index(drop=True)


def rate_curve(df: pd.DataFrame, t_start: int = T_START) -> pd.DataFrame:
    steps = df["step"].to_numpy(dtype=float)
    dist = df["value"].to_numpy(dtype=float)
    mask = steps >= t_start
    v = geometric_absorption(steps[mask], dist[mask], t_start, 1.4)
    return pd.DataFrame({"step": steps[mask], "v": v}).dropna().reset_index(drop=True)


FIG9 = {
    "00-3e-75": "Full-Data (17k)",
    "00-6c-77": "16-Shot",
    "6b-4b-89": "4-Shot",
    "a6-18-2d": "1-Shot",
}


def paper_tail_fits(style) -> None:
    theme = style.apply_rc("tableau-paper")
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), squeeze=False)
    for idx, (hexc, label) in enumerate(FIG9.items()):
        ax = axes[idx // 2][idx % 2]
        d = load_concat(f"5_digestion_rate_v2__panel0__{hexc}*.csv")
        d = d[d["step"] >= T_START]
        t = d["step"].to_numpy(dtype=float)
        y = d["value"].to_numpy(dtype=float)
        ax.plot(
            t,
            y,
            color=style.NEUTRALS["ink"],
            linewidth=1.3,
            label="digitized $d_t/d_{30}$ (L1)",
        )
        colors = style.series_colors("tableau-paper", 4)
        fits = sorted(
            [f(t, y) for f in (fit_power_law, fit_stretched_exp, fit_floor_exp)],
            key=lambda f: f["rmse_log"],
        )
        names = {
            "power_law": "power law",
            "stretched_exp": "stretched exp",
            "floor_exp": "floor+exp",
        }
        for k, fit in enumerate(fits):
            ax.plot(
                t,
                fit["pred"],
                color=colors[k + 1],
                linewidth=1.4,
                linestyle="--",
                label=f"{names[fit['model']]} ({fit['rmse_log']:.3f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([30, 100, 200, 300])
        ax.set_xticklabels(["30", "100", "200", "300"])
        ax.minorticks_off()
        ax.set_title(label, fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("$d_t/d_{30}$")
        if idx >= 2:
            ax.set_xlabel("Training Step (log)")
        ax.legend(loc="lower left", fontsize=6.5)
        style.apply_frame(ax, theme.frame)
        style.apply_grid(ax, theme.grid, axis="both")
    fig.suptitle(
        "M1 on the paper's digitized Figure-9 curves (L1 metric): tail-form fits",
        y=0.995,
    )
    fig.tight_layout()
    out = figure_path("paper", "tail-distance", "step", "source-paper", "fits")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def alpha_comparison(style) -> None:
    theme = style.apply_rc("tableau-paper")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = style.series_colors("tableau-paper", 12)
    idx = 0
    for hexc, label in FIG9.items():
        d = load_concat(f"5_digestion_rate_v2__panel0__{hexc}*.csv")
        d = d[d["step"] >= T_START]
        t = d["step"].to_numpy(dtype=float)
        a = local_alpha(t, d["value"].to_numpy(dtype=float))
        ax.plot(t, a, color=colors[idx], linewidth=1.5, label=f"paper {label} (L1)")
        idx += 1
    for csv_path in sorted(
        (HERE / "derived").glob("*__local_alpha.csv"),
        key=lambda p: "top16" if "be00rj9e" in p.name else "z",
    ):
        run_id = csv_path.name.split("__")[0]
        label = "ours top16 (KL)" if run_id == "be00rj9e" else "ours sampled-token (KL)"
        df = pd.read_csv(csv_path)
        ax.plot(
            df["step"],
            df["local_alpha"],
            color=colors[idx],
            linewidth=1.8,
            linestyle="--",
            label=label,
        )
        idx += 1
    ax.axhline(1.0, color=style.NEUTRALS["baseline"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"local exponent $\alpha(t)$")
    ax.set_title(r"$\alpha(t)$: paper L1 metric (solid) vs our KL proxy (dashed)")
    style.apply_frame(ax, theme.frame)
    style.apply_grid(ax, theme.grid, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5)
    fig.tight_layout()
    out = figure_path("paper", "local-alpha", "step", "source-paper-vs-ours")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def hump_check(style) -> None:
    theme = style.apply_rc("tableau-paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # (a) lr folding: rate vs t for the three arms (H3a)
    colors = style.series_colors("tableau-paper", 3)
    for k, (hexc, lr) in enumerate(
        zip(
            ("9a-67-00", "00-6c-77", "00-3e-75"),
            ("0.5e-6", "1e-6", "2e-6"),
            strict=False,
        )
    ):
        d = load_main_branch(f"5A_lr_collapse_v2__panel1__{hexc}*.csv")
        rc = rate_curve(d)
        ax1.plot(
            rc["step"], rc["v"] * 100, color=colors[k], linewidth=1.6, label=f"lr={lr}"
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel(r"$v_t$ (%)")
    ax1.set_title("lr arms: hump anchored in steps, not in $u$ (H3a)", fontsize=10)
    ax1.legend(fontsize=8)
    style.apply_frame(ax1, theme.frame)
    style.apply_grid(ax1, theme.grid, axis="y")

    # (b) online vs off-policy (H3b)
    colors = style.series_colors("tableau-paper", 4)
    for k, (hexc, label) in enumerate(
        (("00-3e-75", "On-Policy"), ("a6-18-2d", "Off-Policy"))
    ):
        a = load_main_branch(f"5_online_offline__panel2__{hexc}*.csv")
        a["value"] = -a["value"]
        rc = rate_curve(a)
        ax2.plot(
            rc["step"],
            rc["v"] * 100,
            color=colors[k + 2],
            linewidth=1.6,
            label=f"{label} (advantage)",
        )
    for hexc, label, ls in (
        ("00-3e-75", "Full-Data (17k)", "-"),
        ("a6-18-2d", "1-Shot", "-"),
    ):
        v = load_concat(f"5_digestion_rate_v2__panel1__{hexc}*.csv")
        ax2.plot(
            v["step"],
            v["value"],
            color=colors[0 if "Full" in label else 1],
            linewidth=1.1,
            alpha=0.55,
            linestyle="--",
            label=f"paper $v_t$: {label}",
        )
    ax2.set_yscale("log")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("rate (%)")
    ax2.set_title("paper $v_t$ vs advantage-rate: on/off-policy (H3b)", fontsize=10)
    ax2.legend(fontsize=7)
    style.apply_frame(ax2, theme.frame)
    style.apply_grid(ax2, theme.grid, axis="y")

    fig.tight_layout()
    out = figure_path("paper", "absorption", "step", "source-paper", "hump-check")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> int:
    style = load_style()
    (HERE / "figures").mkdir(exist_ok=True)
    paper_tail_fits(style)
    alpha_comparison(style)
    hump_check(style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
