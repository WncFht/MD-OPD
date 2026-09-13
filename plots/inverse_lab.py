#!/usr/bin/env python3
"""Inverse-theory lab for the OPD spectral framework.

Companion numerical experiments to docs/2026-09-13-opd-spectral-inverse-primer.md.
The OPD absorption model d(t) = sum_k c_k exp(-rho_k (t-T0)) is a discrete
Fredholm equation of the first kind with kernel exp(-rho*t) -- the inverse
Laplace transform. This script quantifies how ill-posed the inversion is and
which parts of the inverted spectrum are actually resolved.

  E1  SVD of the design matrix A (real window t in [30,279], 64 log-rho grid):
      singular-value decay, condition number vs grid density.
  E2  Discrete Picard plot on the real d(t) curve (relative-residual weighting
      X = A/y, b = 1, matching the log-space loss the NNLS approximates).
  E3  Model resolution matrix R_m = V_p V_p^T: heatmap + Backus-Gilbert style
      averaging-kernel rows; shows sub-resolution mass leaking to rho_min.
  E4  Synthetic forward->invert: known bimodal spectrum + a DC floor mode
      (timescale >> window); recover with NNLS vs TSVD vs Tikhonov+GCV.
  E5  Filter factors on one sigma axis: TSVD / Tikhonov / Landweber iteration
      (the MDPO inner loop is a Landweber/Neumann truncation of B^{-1} g).

Outputs:
  docs/figures/2026-09-13-opd-spectral-inverse-primer/fig{1..5}_*.png
  plots/derived/inverse_lab.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from plots.core.plotting import load_plot_style  # noqa: E402
import style_presets  # noqa: E402

T0 = 30
RHO_MAX = 0.10
N_GRID = 64
RUN_CSV = HERE / "derived" / "mdxt04lm__alignment.csv"
FIG_DIR = HERE.parent / "docs" / "figures" / "2026-09-13-opd-spectral-inverse-primer"
OUT_JSON = HERE / "derived" / "inverse_lab.json"

rng = np.random.default_rng(7)


def nnls_multiplicative(A: np.ndarray, b: np.ndarray, iters: int = 6000) -> np.ndarray:
    """min_c ||A c - b||^2 s.t. c >= 0 via Lee-Seung multiplicative updates."""
    AtA = A.T @ A
    Atb = A.T @ b
    c = np.full(A.shape[1], b.max() * 0.01)
    for _ in range(iters):
        c *= Atb / (AtA @ c + 1e-300)
    return c


def tikhonov_gcv(s: np.ndarray, Utb: np.ndarray, n: int, lambdas: np.ndarray) -> float:
    """Pick lambda minimizing the GCV score V(lambda) = ||Xc-b||^2 / tr(I-H)^2."""
    best, best_lam = np.inf, lambdas[0]
    for lam in lambdas:
        f = s**2 / (s**2 + lam**2)  # filter factors
        resid2 = np.sum(((1 - f) * Utb) ** 2)
        dof = n - f.sum()  # tr(I - H_lambda)
        score = resid2 / dof**2
        if score < best:
            best, best_lam = score, lam
    return best_lam


def tikhonov_morozov(s: np.ndarray, Utb: np.ndarray, target_resid2: float) -> float:
    """Discrepancy principle: smallest lambda with ||Xc-b||^2 >= noise level."""
    for lam in np.geomspace(s[-1], s[0], 400):
        f = s**2 / (s**2 + lam**2)
        if np.sum(((1 - f) * Utb) ** 2) >= target_resid2:
            return lam
    return s[-1]


def picard_truncation(s: np.ndarray, Utb: np.ndarray) -> tuple[int, float]:
    """Noise-level truncation: keep modes whose sigma_i exceeds the noise
    plateau of |u_i^T b| (estimated from the tail of the coefficients)."""
    plateau = float(np.median(np.abs(Utb)[-12:]))
    p = int(np.sum(s > plateau))
    return max(p, 1), plateau


def main() -> int:
    style = load_plot_style()
    theme = style.apply_rc("tableau-paper")
    pal = style.series_colors("tableau-paper", 5)
    ls = style.line_styles(theme.line_preset)
    muted = style.NEUTRALS["muted"]
    ink = style.NEUTRALS["ink"]
    base = style.NEUTRALS["baseline"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    # ---- data and design matrix ---------------------------------------------
    df = pd.read_csv(RUN_CSV).sort_values("step")
    df = df[df["step"] >= T0]
    t = df["step"].to_numpy(float)
    y = df["d_nats"].to_numpy(float)
    t_end = t.max()
    rho = np.geomspace(1.0 / (t_end - T0), RHO_MAX, N_GRID)
    dln = np.log(rho[1] / rho[0])
    A = np.exp(-np.outer(t - T0, rho))

    # relative-residual formulation: log-space fit ~ min ||(A c)/y - 1||
    X = A / y[:, None]
    b = np.ones_like(y)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    Utb = U.T @ b

    # ---- fig0: forward/inverse chain schematic -------------------------------
    demo_rho = np.array([0.006, 0.02, 0.06])
    demo_c = np.array([0.05, 0.03, 0.02])
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    cols = [pal[0], pal[1], pal[4]]
    dsum = np.zeros_like(t)
    for r_, c_, col in zip(demo_rho, demo_c, cols):
        mode = c_ * np.exp(-r_ * (t - T0))
        dsum += mode
        axL.plot(t, mode, color=col, linewidth=1.4, label=f"$c_k e^{{-{r_:.3f}(t-T_0)}}$")
    axL.plot(t, dsum, color=ink, linewidth=2.0, linestyle=ls["reference"], label="$d(t)=\\sum_k c_k e^{-\\rho_k t}$")
    axL.set_xlabel("step $t$")
    axL.set_ylabel("distance")
    axL.set_title("(a) forward: modes sum to the observed curve", fontsize=10)
    axL.legend(fontsize=7.5)
    style.apply_frame(axL, theme.frame)
    style.apply_grid(axL, theme.grid, axis="both")

    axR.stem(demo_rho, demo_c, linefmt=ink, markerfmt="o", basefmt=" ")
    axR.set_xscale("log")
    axR.set_xticks(demo_rho)
    axR.set_xticklabels([f"{r:g}" for r in demo_rho])
    axR.minorticks_off()
    axR.set_xlabel(r"rate $\rho$")
    axR.set_ylabel(r"mass $c_k$")
    axR.set_title("(b) inverse: recover the rate spectrum", fontsize=10)
    style.apply_frame(axR, theme.frame)
    style.apply_grid(axR, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_chain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- E1: singular-value decay -------------------------------------------
    cond = s[0] / s[-1]
    ranks = {f"rank@1e-{e}": int((s > s[0] * 10**-e).sum()) for e in (2, 4, 6, 8)}
    stable_rank = float(np.sum(s**2) / s[0] ** 2)
    # condition number vs grid density (same window, same rho range)
    cond_sweep = {}
    for ng in (16, 32, 64, 128, 256):
        g = np.geomspace(1.0 / (t_end - T0), RHO_MAX, ng)
        sg = np.linalg.svd(np.exp(-np.outer(t - T0, g)) / y[:, None], compute_uv=False)
        cond_sweep[ng] = float(sg[0] / sg[-1])
    metrics["E1"] = {
        "cond_design": cond,
        "s_max": float(s[0]),
        "s_min": float(s[-1]),
        **ranks,
        "stable_rank": stable_rank,
        "cond_vs_grid": cond_sweep,
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    # kernel library: a few columns of A
    show_idx = [0, 4, 8, 12, 20, 32, 48, 63]
    cmap = plt.get_cmap("viridis")
    for j in show_idx:
        frac = j / (N_GRID - 1)
        axL.plot(t, A[:, j], color=cmap(frac), linewidth=1.5, label=f"$\\rho={rho[j]:.4f}$")
    axL.set_yscale("log")
    axL.set_xlabel("step $t$")
    axL.set_ylabel(r"$e^{-\rho_k\,(t-T_0)}$")
    axL.set_title("(a) kernel library on the observation window", fontsize=10)
    axL.legend(fontsize=6.5, ncol=2)
    style.apply_frame(axL, theme.frame)
    style.apply_grid(axL, theme.grid, axis="both")

    axR.semilogy(np.arange(1, N_GRID + 1), s, "o-", color=pal[0], markersize=3.5, linewidth=1.4)
    for e, col in ((4, pal[1]), (8, pal[3])):
        thr = s[0] * 10**-e
        axR.axhline(thr, color=col, linewidth=0.9, linestyle=ls["reference"], alpha=0.8)
        axR.annotate(
            f"$\\sigma_1\\cdot10^{{-{e}}}$ → rank {ranks[f'rank@1e-{e}']}",
            xy=(N_GRID * 0.55, thr * 2.2), fontsize=7, color=col,
        )
    axR.set_xlabel("singular index $i$")
    axR.set_ylabel(r"$\sigma_i$")
    axR.set_title(f"(b) singular values, relative-residual weighting — $\\kappa\\gtrsim10^{{16}}$", fontsize=10)
    style.apply_frame(axR, theme.frame)
    style.apply_grid(axR, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_kernel_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- E2: discrete Picard plot --------------------------------------------
    ratio = np.abs(Utb) / s
    noise_floor = 0.064  # log-space rmse of the NNLS fit == relative noise level
    p_picard, plateau = picard_truncation(s, Utb)
    metrics["E2"] = {
        "picard_plateau": plateau,
        "noise_floor_rel": noise_floor,
        "p_picard": p_picard,
        "first_15_utb": [float(v) for v in np.abs(Utb)[:15]],
        "ratio_at_p": float(ratio[p_picard - 1]),
        "ratio_at_64": float(ratio[-1]),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    idx = np.arange(1, N_GRID + 1)
    ax.semilogy(idx, s, "o-", color=muted, markersize=3, linewidth=1.2, label=r"$\sigma_i$")
    ax.semilogy(idx, np.abs(Utb), "s-", color=pal[0], markersize=3, linewidth=1.2, label=r"$|u_i^\top b|$")
    ax.semilogy(idx, ratio, "^-", color=pal[3], markersize=3, linewidth=1.2, label=r"$|u_i^\top b|/\sigma_i$")
    ax.axhline(plateau, color=pal[0], linewidth=0.8, linestyle=":", alpha=0.8)
    ax.annotate(f"noise plateau $\\approx{plateau:.3f}$", xy=(38, plateau * 0.28), fontsize=7.5, color=pal[0])
    ax.axvline(p_picard, color=base, linewidth=0.9, linestyle=ls["reference"])
    ax.annotate(f"$p^*={p_picard}$", xy=(p_picard + 1.5, s[0] * 0.02), fontsize=8, color=ink)
    ax.set_xlabel("singular index $i$")
    ax.set_title("discrete Picard plot — real $d_t$ on the 64-mode $\\rho$ grid", fontsize=10)
    ax.legend(fontsize=8)
    style.apply_frame(ax, theme.frame)
    style.apply_grid(ax, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_picard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- E3: resolution matrix / averaging kernels ---------------------------
    p_star = p_picard
    Rm = V[:, :p_star] @ V[:, :p_star].T
    metrics["E3"] = {
        "p_star": p_star,
        "Rm_diag_min": float(np.diag(Rm).min()),
        "Rm_diag_at_rho_min": float(Rm[0, 0]),
        "Rm_diag_at_rho_max": float(Rm[-1, -1]),
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    im = axL.imshow(Rm, origin="lower", extent=[np.log10(rho[0]), np.log10(rho[-1])] * 2,
                    cmap="magma", aspect="auto", vmin=0, vmax=1)
    axL.plot(np.log10(rho), np.log10(rho), color="white", linewidth=0.8, linestyle="--", alpha=0.8)
    axL.set_xlabel(r"$\log_{10}\rho$ (estimated bin)")
    axL.set_ylabel(r"$\log_{10}\rho$ (true bin)")
    axL.set_title(f"(a) $R_m=V_{{{p_star}}}V_{{{p_star}}}^{{\\top}}$", fontsize=10)
    fig.colorbar(im, ax=axL, shrink=0.85)

    for j, col in ((4, pal[0]), (20, pal[1]), (40, pal[4]), (56, pal[3])):
        axR.plot(rho, Rm[:, j], color=col, linewidth=1.5, label=f"kernel at $\\rho={rho[j]:.4f}$")
        axR.axvline(rho[j], color=col, linewidth=0.7, linestyle=":", alpha=0.6)
    axR.set_xscale("log")
    axR.set_xlabel(r"$\rho$")
    axR.set_ylabel(r"$R_m$ column (averaging kernel)")
    axR.set_title("(b) what each estimated bin actually averages", fontsize=10)
    axR.legend(fontsize=7.5)
    style.apply_frame(axR, theme.frame)
    style.apply_grid(axR, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_resolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- E4: synthetic forward->invert ---------------------------------------
    def lognorm_c(center, width, amp):
        return amp * np.exp(-0.5 * (np.log(rho / center) / width) ** 2)

    c_true = (
        lognorm_c(0.05, 0.12, 0.010)      # sharp resolved bump
        + lognorm_c(0.012, 0.30, 0.007)   # broad mid bump
        + lognorm_c(rho[0], 0.5, 0.004)   # slow mass straddling resolution edge
    )
    A_aug = np.column_stack([np.ones_like(t), A])       # true model has a DC mode rho->0
    c_true_aug = np.concatenate([[0.004], c_true])      # floor: timescale >> window
    y_true = A_aug @ c_true_aug
    eps = 0.05
    y_obs = y_true * np.exp(eps * rng.standard_normal(t.size))

    Xs = A / y_obs[:, None]
    bs = np.ones_like(y_obs)
    Us, ss, Vts = np.linalg.svd(Xs, full_matrices=False)
    Vs = Vts.T
    Utbs = Us.T @ bs

    c_nn = nnls_multiplicative(Xs, bs)
    # TSVD at the noise-level truncation
    p_s, plateau_s = picard_truncation(ss, Utbs)
    c_tsvd = Vs[:, :p_s] @ (Utbs[:p_s] / ss[:p_s])
    # Tikhonov via Morozov discrepancy principle (noise level known here)
    lam_mor = tikhonov_morozov(ss, Utbs, t.size * eps**2)
    lam_gcv = tikhonov_gcv(ss, Utbs, t.size, np.geomspace(ss[-1], ss[0], 200))
    f_tik = ss**2 / (ss**2 + lam_mor**2)
    c_tik = Vs @ (f_tik * Utbs / ss)
    # naive full inverse (what raw pseudoinverse does with kappa ~ 1e18)
    c_naive = Vs @ (Utbs / ss)

    res = {"rho_min": rho[0]}
    for name_, c_ in (("nnls", c_nn), ("tsvd", c_tsvd), ("tikhonov", c_tik)):
        res[name_] = {
            "rel_l2": float(np.linalg.norm(c_ - c_true) / np.linalg.norm(c_true)),
            "mass_below_0.01": float(c_[rho < 0.01].sum()),
            "mass_at_rho_min_bin": float(c_[0]),
        }
    res["true_mass_below_0.01"] = float(c_true[rho < 0.01].sum() + c_true_aug[0])
    res["naive_l2"] = float(np.linalg.norm(c_naive))
    res["p_synth"] = p_s
    res["lam_morozov"] = float(lam_mor)
    res["lam_gcv"] = float(lam_gcv)
    metrics["E4"] = res

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axL.plot(t, y_true, color=ink, linewidth=1.4, label="clean $A c_{\\rm true}$")
    axL.plot(t, y_obs, ".", color=pal[0], markersize=2.5, alpha=0.6, label="observed (5% noise)")
    axL.plot(t, A @ c_nn, color=pal[1], linewidth=1.2, linestyle=ls["reference"], label="NNLS fit")
    axL.set_yscale("log")
    axL.set_xlabel("step $t$")
    axL.set_ylabel("$d(t)$")
    axL.set_title("(a) forward model + 5% noise", fontsize=10)
    axL.legend(fontsize=7.5)
    style.apply_frame(axL, theme.frame)
    style.apply_grid(axL, theme.grid, axis="both")

    # mass per grid bin, linear y: TSVD sign-oscillation speaks for itself
    axR.plot(rho, c_true, color=ink, linewidth=1.8, label="true $c_k$ (+ off-grid DC floor)")
    axR.plot(rho, c_nn, "o-", color=pal[1], markersize=3, linewidth=1.1, label="NNLS")
    axR.plot(rho, c_tik, "^:", color=pal[4], markersize=3, linewidth=1.1, label=f"Tikhonov Morozov $\\lambda={lam_mor:.0f}$")
    axR.plot(rho, c_tsvd, "s--", color=pal[0], markersize=2.5, linewidth=0.9, alpha=0.85,
             label=f"TSVD $p={p_s}$ (sign-oscillating)")
    axR.set_xscale("log")
    axR.set_ylim(-0.03, 0.05)
    axR.axvline(1.0 / (t_end - T0), color=base, linewidth=0.9, linestyle=ls["reference"], alpha=0.8)
    axR.annotate(r"$\rho=1/(T{-}T_0)$", xy=(1.06 / (t_end - T0), 0.045), fontsize=7, color=base)
    axR.set_xlabel(r"$\rho$")
    axR.set_ylabel(r"mass per bin $c_k$")
    axR.set_title("(b) recovered vs true spectrum", fontsize=10)
    axR.legend(fontsize=7.5, loc="upper right")
    style.apply_frame(axR, theme.frame)
    style.apply_grid(axR, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_synthetic_inversion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- E5: filter-factor family on one sigma axis ---------------------------
    f_tsvd = (np.arange(N_GRID) < p_s).astype(float)
    f_tik_real = ss**2 / (ss**2 + lam_mor**2)
    a_land = 1.0 / ss[0] ** 2  # Landweber step, stability alpha*sigma^2 < 2
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axL.semilogx(ss / ss[0], f_tsvd, "o-", color=pal[0], markersize=3, linewidth=1.2, label=f"TSVD $p={p_s}$ (hard cut)")
    axL.semilogx(ss / ss[0], f_tik_real, "s-", color=pal[4], markersize=3, linewidth=1.2, label="Tikhonov (Morozov $\\lambda$)")
    for m, col in ((4, pal[1]), (16, pal[2]), (64, pal[3])):
        f_lw = 1 - (1 - a_land * ss**2) ** m
        axL.semilogx(ss / ss[0], f_lw, "-", color=col, linewidth=1.4, label=f"Landweber $m={m}$")
    axL.set_xlabel(r"singular value $\sigma_i/\sigma_1$")
    axL.set_ylabel("filter factor $f_i$")
    axL.set_title("(a) three regularizers on one $\\sigma$ axis", fontsize=10)
    axL.legend(fontsize=7.5)
    style.apply_frame(axL, theme.frame)
    style.apply_grid(axL, theme.grid, axis="both")

    # right panel: MDPO/Neumann delivery vs alpha*lambda (stability picture)
    x = np.linspace(0, 2.6, 600)
    for m, col in ((1, muted), (4, pal[1]), (16, pal[2]), (64, pal[3])):
        deliv = 1 - (1 - x) ** m
        axR.plot(x, deliv, color=col, linewidth=1.4, label=f"$m={m}$")
    axR.axvline(2.0, color=pal[3], linewidth=0.9, linestyle=ls["reference"])
    axR.annotate(r"$\alpha\lambda=2$: GD stability edge", xy=(2.02, 2.6), fontsize=7.5, rotation=90, va="top", color=pal[3])
    axR.axhline(1.0, color=base, linewidth=0.8, linestyle=":", alpha=0.7)
    axR.set_xlabel(r"$\alpha\lambda$  (mode rate per inner step)")
    axR.set_ylabel(r"delivered fraction $1-(1-\alpha\lambda)^m$")
    axR.set_ylim(-0.2, 3.2)
    axR.set_title("(b) Neumann-series view of the MDPO inner loop", fontsize=10)
    axR.legend(fontsize=7.5)
    style.apply_frame(axR, theme.frame)
    style.apply_grid(axR, theme.grid, axis="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_filter_factors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- real-data inversion crosscheck: mass below resolution ---------------
    c_real = nnls_multiplicative(X, b)
    metrics["E6_real"] = {
        "mass_total": float(c_real.sum()),
        "mass_below_0.01": float(c_real[rho < 0.01].sum()),
        "frac_below_0.01": float(c_real[rho < 0.01].sum() / c_real.sum()),
        "mass_at_rho_min_bin": float(c_real[0]),
        "fit_log_rmse": float(np.sqrt(np.mean((np.log(np.maximum(A @ c_real, 1e-300)) - np.log(y)) ** 2))),
    }

    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"[{k}] {json.dumps(v)[:400]}")
    print(f"figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
