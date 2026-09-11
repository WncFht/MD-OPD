"""Run every lr-ablation figure module in one command."""

from . import absorption, advantage, alpha, baselines, eval_fit, panels, supplement, topk


def main() -> None:
    """Render all lr-ablation figures."""
    baselines.main()
    panels.main()
    supplement.d1()
    supplement.d2()
    supplement.d5()
    print(eval_fit.main())
    print(absorption.plot_absorption_rate())
    alpha.main()
    print(advantage.plot_advantage_diagnostics())
    print(topk.plot_topk_diagnostics())


if __name__ == "__main__":
    main()
