from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from montecarlo_trading.engine import MarketModel, SimulationConfig, run_experiment
from montecarlo_trading.strategies import FixedFraction, Kelly, FractionalKelly, CappedKelly
from montecarlo_trading.plots import plot_terminal_histograms, plot_sample_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.53)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    model = MarketModel(p=args.p, b=args.b)
    cfg = SimulationConfig(rounds=args.rounds, trials=args.trials, initial_wealth=1.0, ruin_threshold=0.2, seed=args.seed)

    strategies = [
        FixedFraction(0.02, name="Fixed_2pct"),
        FixedFraction(0.05, name="Fixed_5pct"),
        Kelly(name="Kelly"),
        FractionalKelly(k=0.5, name="HalfKelly"),
        CappedKelly(cap=0.10, name="KellyCapped_10pct"),
    ]

    metrics_by_name, terminal_by_name, paths_by_name = run_experiment(model, cfg, strategies)

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Save summary
    rows = []
    for name, m in metrics_by_name.items():
        row = {
            "strategy": name,
            "n": m.n,
            "terminal_mean": m.terminal_mean,
            "terminal_mean_ci_low": m.ci["terminal_mean"][0],
            "terminal_mean_ci_high": m.ci["terminal_mean"][1],
            "terminal_median": m.terminal_median,
            "terminal_p10": m.terminal_p10,
            "terminal_p90": m.terminal_p90,
            "loss_probability": m.loss_probability,
            "loss_prob_ci_low": m.ci["loss_probability"][0],
            "loss_prob_ci_high": m.ci["loss_probability"][1],
            "ruin_probability": m.ruin_probability,
            "ruin_prob_ci_low": m.ci["ruin_probability"][0],
            "ruin_prob_ci_high": m.ci["ruin_probability"][1],
            "mean_log_growth": m.mean_log_growth,
            "mean_log_growth_ci_low": m.ci["mean_log_growth"][0],
            "mean_log_growth_ci_high": m.ci["mean_log_growth"][1],
            "vol_log_growth": m.vol_log_return,
            "max_drawdown_mean_sample": m.max_drawdown_mean,
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(by="mean_log_growth", ascending=False)
    df.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    # Save terminals + a few paths
    for name, W in terminal_by_name.items():
        pd.DataFrame({"terminal_wealth": W}).to_csv(os.path.join(args.outdir, f"terminal_{name}.csv"), index=False)
    for name, paths in paths_by_name.items():
        if paths is None:
            continue
        pd.DataFrame(paths).to_csv(os.path.join(args.outdir, f"paths_{name}.csv"), index=False)

    # Plots
    plot_terminal_histograms(terminal_by_name, plots_dir)
    plot_sample_paths(paths_by_name, plots_dir)

    print("Wrote outputs to:", args.outdir)
    print(df)


if __name__ == "__main__":
    main()
