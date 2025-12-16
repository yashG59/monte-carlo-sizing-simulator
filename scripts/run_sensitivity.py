from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from montecarlo_trading.engine import MarketModel, SimulationConfig, run_experiment
from montecarlo_trading.strategies import FixedFraction, Kelly, FractionalKelly, CappedKelly, kelly_fraction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="outputs_sensitivity")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ps = [0.51, 0.53, 0.55]
    bs = [0.8, 1.0, 1.2]

    rows = []
    for p in ps:
        for b in bs:
            model = MarketModel(p=p, b=b)
            cfg = SimulationConfig(rounds=args.rounds, trials=args.trials, initial_wealth=1.0, ruin_threshold=0.2, seed=args.seed)
            kf = kelly_fraction(p, b)
            strategies = [
                FixedFraction(0.02, name="Fixed_2pct"),
                FixedFraction(0.05, name="Fixed_5pct"),
                FractionalKelly(k=0.5, name="HalfKelly"),
                Kelly(name="Kelly"),
                CappedKelly(cap=0.10, name="KellyCapped_10pct"),
            ]
            metrics_by_name, _, _ = run_experiment(model, cfg, strategies, ci_bootstrap_samples=400)

            for name, m in metrics_by_name.items():
                rows.append({
                    "p": p, "b": b, "kelly_f": kf,
                    "strategy": name,
                    "mean_log_growth": m.mean_log_growth,
                    "ruin_probability": m.ruin_probability,
                    "loss_probability": m.loss_probability,
                    "terminal_mean": m.terminal_mean,
                    "terminal_median": m.terminal_median,
                })

    df = pd.DataFrame(rows).sort_values(["p", "b", "mean_log_growth"], ascending=[True, True, False])
    df.to_csv(os.path.join(args.outdir, "sensitivity_summary.csv"), index=False)
    print("Wrote:", os.path.join(args.outdir, "sensitivity_summary.csv"))
    print(df.head(20))


if __name__ == "__main__":
    main()
