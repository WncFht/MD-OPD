"""Named experiment groups used by analysis entry points."""

M4_RUNS = {
    "ybi7q89v": {"label": r"lr $5\times10^{-7}$", "scale": 0.5, "color": "#b07a00"},
    "mdxt04lm": {"label": r"lr $1\times10^{-6}$", "scale": 1.0, "color": "#1f8a9a"},
    "yj2w13fq": {"label": r"lr $2\times10^{-6}$", "scale": 2.0, "color": "#245b9a"},
    "ml6mz58f": {"label": r"lr $5\times10^{-6}$", "scale": 5.0, "color": "#c44e52"},
    "l0a6rsc1": {"label": r"lr $1\times10^{-5}$", "scale": 10.0, "color": "#8c564b"},
    "tjjxnvw9": {"label": r"$\beta_2=0.95$", "scale": 1.0, "color": "#7a3e9d"},
}


def m4_run_ids() -> list[str]:
    """Return M4 run ids in the canonical plotting order."""
    return list(M4_RUNS)
