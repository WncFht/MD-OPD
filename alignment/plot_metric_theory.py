#!/usr/bin/env python3
"""Absorption-metric theory figures: what d_t measures, and what to log instead.

Theory panels (synthetic, from the spectral model in
docs/2026-09-06-absorption-rate-spectral-theory*.md):
    T1  single-observable degeneracy: two different spectra, one d(t) curve
    T2  calibers are observation vectors: cross-caliber slope ratio S(t) >= 2
    T3  position / movement / efficiency triple separates converged / spinning / silent

Data panels (alignment/runs/*.csv, pulled by pull_wandb_metrics.py):
    D1  KL proxy = CE - H: subtraction of two slow variables
    D2  mean collapse: heavy tail of |A| vs its mean
    D3  downstream eval vs ln d: d as a sufficient statistic, with task-dependent slope
    D4  three empirically distinct "flat d" regimes
    D5  optimizer sensitivity of v_t (M4 partial arms)

Style: .agents/skills/plot-skill (tableau-paper preset).
Output: alignment/figures/metric/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "plot-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

import matplotlib.pyplot as plt  # noqa: E402
from style_presets import (  # noqa: E402
    NEUTRALS,
    apply_frame,
    apply_grid,
    apply_rc,
    line_styles,
    series_colors,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "alignment" / "runs"
OUT = ROOT / "alignment" / "figures" / "metric"
OUT.mkdir(parents=True, exist_ok=True)

THEME = "tableau-paper"


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def load_run(run_id: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / f"{run_id}__history.csv").sort_values("_step")
    df["d"] = -df["opd/token_reward_mean"]
    df["ce"] = -df["opd/teacher_logp_mean"]
    return df


def v_geo(d: np.ndarray, steps: np.ndarray, t: float, t0: int = 30) -> float | None:
    """Paper-style geometric-window absorption rate: OLS slope of ln d on [t/1.4, 1.4t].

    The window must fit entirely inside [t0, T] (paper appendix C.1 convention);
    a truncated window biases the tail estimate, so out-of-range t returns None.
    """
    lo, hi = int(np.ceil(t / 1.4)), int(np.floor(1.4 * t))
    if lo < t0 or hi > int(steps.max()):
        return None
    m = (steps >= lo) & (steps <= hi)
    if m.sum() < 5:
        return None
    beta = np.polyfit(steps[m].astype(float), np.log(d[m]), 1)[0]
    return 1 - np.exp(beta)


# ---------------------------------------------------------------------------
# T1: single-observable degeneracy
# ---------------------------------------------------------------------------


def theory_t1() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 2)

    # spectrum A: discrete scale-free cascade, rho_i = rho1 i^-beta, a_i ∝ i^-delta
    i = np.arange(1, 4001)
    rho_a = 0.2 * i ** (-1.0)
    a_a = i ** (-2.0)
    a_a /= a_a.sum()

    def d_power(t: np.ndarray) -> np.ndarray:
        return (a_a[:, None] * np.exp(-rho_a[:, None] * t[None, :])).sum(0)

    # spectrum B: continuous Gamma-density weights (finite band edge), closed-form d
    gamma_b, c_b = 1.02, 1.2226

    def d_gamma(t: np.ndarray) -> np.ndarray:
        return (t + c_b) ** (-gamma_b)

    t_fit = np.arange(30.0, 301.0)  # the measurement window the claim is about
    raw = np.log(d_gamma(t_fit)) - np.log(d_power(t_fit))
    offset = float(raw.mean())
    rmse = float(np.sqrt(((raw - offset) ** 2).mean()))

    t_all = np.geomspace(30, 3000, 200)
    dA = d_power(t_all)
    dB = d_gamma(t_all) * np.exp(-offset)  # align levels the way a fit would

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax1.fill_between(
        t_all,
        (dA / dA[0]) * 1.02,
        (dA / dA[0]) * 0.98,
        color=NEUTRALS["soft_fill"],
        zorder=0,
        label="±2% single-run step noise",
    )
    ax1.loglog(
        t_all,
        dA / dA[0],
        color=colors[0],
        lw=2.0,
        label="spectrum A: discrete power-law cascade",
    )
    ax1.loglog(
        t_all,
        dB / dA[0],
        color=colors[1],
        lw=2.0,
        ls=styles["reference"],
        label="spectrum B: continuous band edge",
    )
    ax1.set_xlabel("step $t$")
    ax1.set_ylabel("normalised distance $d(t)/d(30)$")
    ax1.set_title(f"one observable, two unrelated spectra (log-rmse {rmse:.4f})")
    apply_frame(ax1, current.frame)
    apply_grid(ax1, current.grid, axis="both")
    ax1.legend(loc="lower left", fontsize=9)

    ax2.loglog(
        rho_a,
        a_a / a_a.max(),
        color=colors[0],
        lw=2.0,
        label="A: $\\rho_i\\propto i^{-1},\\ a_i\\propto i^{-2}$",
    )
    rho_grid = np.geomspace(rho_a.min(), rho_a.max(), 400)
    w_b = rho_grid ** (gamma_b - 1) * np.exp(-c_b * rho_grid)
    ax2.loglog(
        rho_grid,
        w_b / w_b.max(),
        color=colors[1],
        lw=2.0,
        ls=styles["reference"],
        label=f"B: $\u03c1^{{\u03b3-1}}e^{{-\u03c1/\u03c1_c}}$, $\u03b3$={gamma_b:.2f}",
    )
    ax2.set_xlabel("per-step rate $\\rho=\\eta\\lambda$")
    ax2.set_ylabel("initial spectral weight $\\nu_0(\\rho)$")
    ax2.set_title("the two underlying spectra share nothing")
    apply_frame(ax2, current.frame)
    apply_grid(ax2, current.grid, axis="both")
    ax2.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    save(fig, "T1_single_microphone.png")


# ---------------------------------------------------------------------------
# T2: calibers are observation vectors; slope ratio S(t) >= 2
# ---------------------------------------------------------------------------


def theory_t2() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 4)

    # wide discrete spectrum, rates spanning ~1 decade (matches the measured range)
    rng = np.random.default_rng(7)
    n_modes = 14
    rho = np.geomspace(3e-3, 3e-2, n_modes)
    a = 1.0 / rho**0.55
    a *= rng.uniform(0.8, 1.2, n_modes)
    a /= a.sum()

    t = np.arange(30.0, 4001.0)
    d_l1 = (a[:, None] * np.exp(-rho[:, None] * t[None, :])).sum(0)  # rate rho
    d_kl = (rho[:, None] * a[:, None] * np.exp(-2 * rho[:, None] * t[None, :])).sum(0)  # rate 2rho, lambda-weighted
    d_rms = np.sqrt((a[:, None] * np.exp(-2 * rho[:, None] * t[None, :])).sum(0))  # rate 2rho, unweighted

    def local_logslope(y: np.ndarray, win: int = 121) -> np.ndarray:
        out = np.full_like(y, np.nan)
        half = win // 2
        for k in range(half, len(y) - half):
            sl = slice(k - half, k + half + 1)
            out[k] = np.polyfit(t[sl], np.log(y[sl]), 1)[0]
        return out

    s_l1, s_kl = local_logslope(d_l1), local_logslope(d_kl)
    S = s_kl / s_l1  # both slopes negative -> S > 0; S = 2<rho>_u/<rho>_w exactly

    # single-mode reference: S = 2 exactly
    d1_l1 = np.exp(-rho[6] * t)
    d1_kl = rho[6] * np.exp(-2 * rho[6] * t)
    s1_l1, s1_kl = local_logslope(d1_l1), local_logslope(d1_kl)
    S1 = s1_kl / s1_l1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax1.loglog(
        t,
        d_l1 / d_l1[0],
        color=colors[0],
        lw=2.0,
        label="$d^{\\mathrm{L1}}=\\mathbb{E}|A|$: rate $\\eta\\lambda$",
    )
    ax1.loglog(
        t,
        d_rms / d_rms[0],
        color=colors[1],
        lw=1.8,
        label="$d^{\\mathrm{RMS}}=\\sqrt{\\mathbb{E}A^2}$: rate $2\\eta\\lambda$",
    )
    ax1.loglog(
        t,
        d_kl / d_kl[0],
        color=colors[2],
        lw=1.8,
        ls=styles["reference"],
        label="$d^{\\mathrm{KL}}$: rate $2\\eta\\lambda$, $\\lambda$-weighted",
    )
    ax1.set_xlabel("step $t$")
    ax1.set_ylabel("normalised distance")
    ax1.set_title("same spectrum, three calibers = three observation vectors")
    apply_frame(ax1, current.frame)
    apply_grid(ax1, current.grid, axis="both")
    ax1.legend(loc="lower left", fontsize=9)

    ax2.plot(
        t[np.isfinite(S)],
        S[np.isfinite(S)],
        color=colors[0],
        lw=2.0,
        label="wide spectrum (two calibers)",
    )
    ax2.plot(
        t[np.isfinite(S1)],
        S1[np.isfinite(S1)],
        color=NEUTRALS["muted"],
        lw=1.6,
        ls=styles["reference"],
        label="single mode: $S\\equiv 2$",
    )
    ax2.set_xlabel("step $t$")
    ax2.set_ylabel(
        "slope ratio  $S(t)=\\dot{\\ln d^{\\mathrm{KL}}}\\,/\\,\\dot{\\ln d^{\\mathrm{L1}}}$"
        "\n"
        r"$=\,2\,E_u[\rho]\,/\,E_w[\rho],\ \ u\propto\rho\,e^{-\rho t}\,w$"
    )
    ax2.set_title("departure from 2 = per-step spectral signature, no fitting")
    apply_frame(ax2, current.frame)
    apply_grid(ax2, current.grid, axis="y")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "T2_caliber_ratio.png")


# ---------------------------------------------------------------------------
# T3: position / movement / efficiency
# ---------------------------------------------------------------------------


def theory_t3() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 3)

    # single mode + OU gradient noise: x_{t+1} = (1-rho) x_t + sqrt(nu) xi
    rho, nu = 0.02, 1e-6
    t = np.arange(30.0, 601.0)
    e = np.exp(-2 * rho * (t - t[0]))
    x2_floor = nu / (2 * rho)  # stationary E[x^2] (OU)
    x2 = x2_floor + 1.0 * e  # decaying signal + stationary floor

    pos = 0.5 * rho * x2  # KL-type position d_t
    move = 0.5 * rho * (rho**2 * x2 + nu)  # KL(pi_t || pi_{t+1}) per step
    dd = rho**2 * x2  # per-step position drop (signal part)

    move_floor = 0.5 * rho * nu  # diffusion-only movement
    # scenario 2: spinning -- large orthogonal parameter motion, d untouched
    pos_spin = np.full_like(t, pos[0])
    move_spin = np.full_like(t, 50.0 * move[0])
    dd_spin = np.zeros_like(t)
    # scenario 3: silent -- no effective step reaches the modes
    pos_silent = np.full_like(t, pos[0])
    move_silent = np.full_like(t, 0.01 * move_floor)
    dd_silent = np.zeros_like(t)

    scenarios = [
        ("converging to the noise floor", pos, move, dd, colors[0]),
        (
            "spinning: movement without absorption",
            pos_spin,
            move_spin,
            dd_spin,
            colors[1],
        ),
        (
            "silent: no signal reaching the modes",
            pos_silent,
            move_silent,
            dd_silent,
            colors[2],
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(11.4, 7.6), sharex="col")
    for col, (name, p, m, drop, c) in enumerate(scenarios):
        axes[0, col].plot(t, p, color=c, lw=2.0)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_ylabel("position $d_t$", fontsize=9)
        axes[1, col].plot(t, m, color=c, lw=2.0)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylabel("movement  $\\mathrm{KL}(\\pi_t\\|\\pi_{t+1})$", fontsize=9)
        axes[2, col].plot(t, drop, color=c, lw=2.0)
        axes[2, col].set_yscale("symlog", linthresh=1e-6)
        axes[2, col].set_ylabel("position drop  $d_t-d_{t+1}$", fontsize=9)
        axes[2, col].set_xlabel("step $t$", fontsize=9)
        for row in range(3):
            apply_frame(axes[row, col], current.frame)
            apply_grid(axes[row, col], current.grid, axis="y")
            axes[row, col].tick_params(labelsize=8)
    fig.suptitle(
        "flat $d_t$ is three different states: position alone cannot separate them — movement and per-step drop do",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "T3_position_movement_efficiency.png")


# ---------------------------------------------------------------------------
# D1: KL proxy = CE - H
# ---------------------------------------------------------------------------


def data_d1() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 3)
    styles = line_styles(current.line_preset)

    df = load_run("mdxt04lm")
    df = df[df["_step"] >= 30].reset_index(drop=True)
    kl = df["ce"] - df["actor/entropy"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax1.plot(
        df["_step"],
        df["ce"],
        color=colors[0],
        lw=2.0,
        label="CE $=-\\,$opd/teacher_logp_mean",
    )
    ax1.plot(
        df["_step"],
        df["actor/entropy"],
        color=colors[1],
        lw=2.0,
        label="student entropy $H$",
    )
    ax1.plot(
        df["_step"],
        kl,
        color=colors[2],
        lw=2.0,
        ls=styles["reference"],
        label="KL proxy $=$ CE $-\\,H$ (= our $d_t$)",
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("nats / token")
    ax1.set_title("run mdxt04lm: the proxy is a difference of two slow variables")
    apply_frame(ax1, current.frame)
    apply_grid(ax1, current.grid, axis="y")
    ax1.legend(loc="upper right", fontsize=9)

    ratio = kl / df["ce"] * 100
    ax2.semilogy(df["_step"], ratio, color=colors[2], lw=2.0)
    ax2.axhline(100, color=NEUTRALS["baseline"], lw=1.0, ls=":")
    ax2.annotate(f"step 30: {ratio.iloc[0]:.0f}%", xy=(32, ratio.iloc[0] * 1.15), fontsize=9)
    ax2.annotate(f"step 278: {ratio.iloc[-1]:.1f}%", xy=(200, ratio.iloc[-1] * 3.2), fontsize=9)
    ax2.set_xlabel("step")
    ax2.set_ylabel("KL proxy as % of CE")
    ax2.set_title("at the tail the signal is <1% of each operand")
    apply_frame(ax2, current.frame)
    apply_grid(ax2, current.grid, axis="y")

    fig.tight_layout()
    save(fig, "D1_ce_h_kl_decomposition.png")


# ---------------------------------------------------------------------------
# D2: mean collapse / heavy tail
# ---------------------------------------------------------------------------


def data_d2() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 3)
    styles = line_styles(current.line_preset)

    df = load_run("mdxt04lm")
    df = df[df["_step"] >= 30].reset_index(drop=True)
    mean_abs = df["d"].abs()
    ext = np.maximum(df["critic/advantages/max"].abs(), df["critic/advantages/min"].abs())
    roll = ext.rolling(21, center=True, min_periods=5).median()
    ratio = ext / mean_abs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax1.semilogy(
        df["_step"],
        mean_abs,
        color=colors[0],
        lw=2.0,
        label="$\\mathbb{E}[A]$ (our $d_t$)",
    )
    ax1.semilogy(df["_step"], ext, color=colors[1], alpha=0.25, lw=0.8)
    ax1.semilogy(
        df["_step"],
        roll,
        color=colors[1],
        lw=2.0,
        label="batch extreme $\\max|A_i|$ (21-step median)",
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("nats / token")
    ax1.set_title("the mean decays 15x; the extreme tail never does")
    apply_frame(ax1, current.frame)
    apply_grid(ax1, current.grid, axis="y")
    ax1.legend(loc="lower left", fontsize=9)

    ax2.semilogy(df["_step"], ratio, color=colors[2], lw=2.0)
    ax2.annotate(f"step 30: {ratio.iloc[0]:.0f}x", xy=(33, ratio.iloc[0] * 1.2), fontsize=9)
    ax2.annotate(f"step 278: {ratio.iloc[-1]:.0f}x", xy=(190, ratio.iloc[-1] * 1.5), fontsize=9)
    ax2.set_xlabel("step")
    ax2.set_ylabel("$\\max|A_i|\\,/\\,\\mathbb{E}|A|$")
    ax2.set_title("a single mean cannot represent this distribution")
    apply_frame(ax2, current.frame)
    apply_grid(ax2, current.grid, axis="y")

    fig.tight_layout()
    save(fig, "D2_mean_collapse_heavy_tail.png")


# ---------------------------------------------------------------------------
# D3: eval vs ln d
# ---------------------------------------------------------------------------


def data_d3() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 3)

    df = load_run("mdxt04lm")
    d = df.set_index("_step")["d"]
    ev = pd.read_csv(RUNS / "mdxt04lm__eval.csv")
    ev["minus_lnd"] = -np.log(np.array([d.loc[s] for s in ev["step"]]))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    benches = [
        ("AIME25", "AIME25", "o", (6, 4)),
        ("AMC23", "AMC23", "^", (6, -13)),
        ("MATH500", "MATH-500 (near ceiling)", "s", (6, 2)),
    ]
    for (col, label, mk, off), c in zip(benches, colors):
        x, y = ev["minus_lnd"].values, ev[col].values * 100
        ax.scatter(x, y, color=c, s=34, marker=mk, label=label, zorder=3)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        pred = np.polyval(coef, x)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.plot(xs, np.polyval(coef, xs), color=c, lw=1.6, ls="--", alpha=0.8)
        ax.annotate(
            f"$R^2={r2:.2f}$",
            xy=(xs[-1], np.polyval(coef, xs[-1])),
            xytext=off,
            textcoords="offset points",
            fontsize=9,
            color=c,
        )
    ax.margins(x=0.10)
    ax.set_xlabel("cumulative absorption  $-\\ln\\, d_t$   (nats/token, rightward = more absorbed)")
    ax.set_ylabel("downstream accuracy (%, mean@16)")
    ax.set_title("run mdxt04lm: ln d tracks capability -- with task-dependent slope")
    apply_frame(ax, current.frame)
    apply_grid(ax, current.grid, axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "D3_eval_vs_logd.png")


# ---------------------------------------------------------------------------
# D4: three flat-d regimes (measured)
# ---------------------------------------------------------------------------


def data_d4() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 3)

    runs = [
        ("mdxt04lm", "healthy: d falls 15x\n(grad_norm 0.08-0.9)", colors[0]),
        (
            "mzwm9q04",
            "spinning: flat d, grad_norm x100\n(discounted-reward misconfig)",
            colors[1],
        ),
        (
            "um5vzc6m",
            "silent: flat d, normal grad_norm\n(Qwen3-1.7B student)",
            colors[2],
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4), sharex="col")
    for col, (rid, label, c) in enumerate(runs):
        df = load_run(rid)
        df = df[df["_step"] >= 10]
        d30 = df[df["_step"] >= 30]["d"].iloc[0]
        ax = axes[0, col]
        ax.plot(df["_step"], df["d"] / d30, color=c, lw=1.8)
        ax.axhline(1.0, color=NEUTRALS["baseline"], lw=0.9, ls=":")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9.5)
        if col == 0:
            ax.set_ylabel("$d_t\\,/\\,d_{30}$", fontsize=9)
        ax2 = axes[1, col]
        ax2.plot(df["_step"], df["actor/grad_norm"], color=c, lw=1.8)
        ax2.set_yscale("log")
        ax2.set_xlabel("step", fontsize=9)
        if col == 0:
            ax2.set_ylabel("actor/grad_norm", fontsize=9)
        for row in range(2):
            apply_frame(axes[row, col], current.frame)
            apply_grid(axes[row, col], current.grid, axis="y")
            axes[row, col].tick_params(labelsize=8)
    fig.suptitle(
        "three empirically distinct regimes that all look like 'd stopped moving' (runs as configured on wandb)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "D4_three_flat_d_regimes.png")


# ---------------------------------------------------------------------------
# D5: optimizer sensitivity of v_t (M4 partial arms)
# ---------------------------------------------------------------------------


def data_d5() -> None:
    current = apply_rc(THEME)
    colors = series_colors(THEME, 5)
    styles = line_styles(current.line_preset)

    def v_curve(rid: str, scale: float = 1.0, tmax: float | None = None):
        df = load_run(rid)
        d, steps = df["d"].values, df["_step"].values
        vs = [(u, v_geo(d, steps, u / scale)) for u in np.arange(42 * scale, (tmax or steps.max()) * scale, 4.0)]
        us = [u for u, v in vs if v is not None]
        return np.array(us), np.array([v for _, v in vs if v is not None]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    # left: beta2 effect at matched optimizer steps
    for rid, label, c, lw in [
        ("mdxt04lm", "baseline $\\beta_2=0.999$", colors[0], 2.0),
        ("ur3j4vb5", "baseline seed 2", colors[1], 1.4),
        ("tjjxnvw9", "$\\beta_2=0.95$ arm", colors[2], 2.0),
    ]:
        u, v = v_curve(rid)
        ax1.plot(u, v, color=c, lw=lw, label=label)
    ax1.annotate(
        "geometric-window $v_t$ exists only up to $t\\leq T/1.4$\n(paper C.1: 218-step run $\\to$ curve ends $\\approx$155)",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        color=NEUTRALS["axis"],
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("$v_t$ (%/step, geometric window)")
    ax1.set_title("beta2 arm vs baseline seeds (matched steps)")
    apply_frame(ax1, current.frame)
    apply_grid(ax1, current.grid, axis="y")
    ax1.legend(loc="upper right", fontsize=9)

    # right: lr folding on u = (lr/1e-6) t
    for rid, scale, label, c in [
        ("yj2w13fq", 2.0, "lr 2e-6", colors[2]),
        ("mdxt04lm", 1.0, "lr 1e-6 (baseline)", colors[0]),
        ("ybi7q89v", 0.5, "lr 5e-7", colors[3]),
    ]:
        u, v = v_curve(rid, scale=scale)
        ax2.plot(u, v, color=c, lw=2.0 if "baseline" in label else 1.7, label=label)
    ax2.set_xlabel("folded axis  $u=(\\mathrm{lr}/10^{-6})\\,t$")
    ax2.set_ylabel("$v_t$ (%/step)")
    ax2.set_title("lr arms collapse on $u$ (partial runs, 1 seed)")
    apply_frame(ax2, current.frame)
    apply_grid(ax2, current.grid, axis="y")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "D5_optimizer_sensitivity.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["t1", "t2", "t3", "d1", "d2", "d3", "d4", "d5"]
    fn = {
        k[0] + k[1] if False else k: f
        for k, f in {
            "t1": theory_t1,
            "t2": theory_t2,
            "t3": theory_t3,
            "d1": data_d1,
            "d2": data_d2,
            "d3": data_d3,
            "d4": data_d4,
            "d5": data_d5,
        }.items()
    }
    for w in which:
        fn[w.lower()]()
