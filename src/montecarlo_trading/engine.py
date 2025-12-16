from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .strategies import Strategy
from .metrics import MetricsResult, compute_metrics


@dataclass(frozen=True)
class MarketModel:
    """
    Binary bet model:
      Win with prob p: wealth *= (1 + f*b)
      Lose with prob 1-p: wealth *= (1 - f)
    """
    p: float  # win probability
    b: float  # payoff multiple on wins (even odds => b=1.0)

    def validate(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError("p must be in (0,1)")
        if self.b <= 0:
            raise ValueError("b must be > 0")


@dataclass(frozen=True)
class SimulationConfig:
    rounds: int = 200
    trials: int = 10000
    initial_wealth: float = 1.0
    ruin_threshold: float = 0.2  # "ruin" if wealth < threshold * initial_wealth
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be > 0")
        if self.trials <= 0:
            raise ValueError("trials must be > 0")
        if self.initial_wealth <= 0:
            raise ValueError("initial_wealth must be > 0")
        if not (0 < self.ruin_threshold < 1):
            raise ValueError("ruin_threshold must be in (0,1)")


def simulate_strategy(
    model: MarketModel,
    cfg: SimulationConfig,
    strategy: Strategy,
    *,
    return_paths: bool = False,
    sample_paths: int = 50,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Simulate a single strategy.
    Returns:
      terminal wealth array shape (trials,)
      optional sample paths array shape (k, rounds+1)
    """
    model.validate()
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)

    # Pre-generate outcomes for speed: True=win, False=loss
    wins = rng.random((cfg.trials, cfg.rounds)) < model.p

    W = np.full((cfg.trials,), cfg.initial_wealth, dtype=np.float64)

    # For drawdown & path samples
    store_paths = return_paths
    k = min(sample_paths, cfg.trials)
    paths = None
    if store_paths:
        # We'll store only k paths for display; full path storage can be huge.
        idx = np.arange(k)
        paths = np.empty((k, cfg.rounds + 1), dtype=np.float64)
        paths[:, 0] = W[idx]

    for t in range(cfg.rounds):
        f = strategy.fraction(model=model, wealth=W, t=t)
        # Clamp to [0, 0.999] to prevent negative wealth from f>=1
        f = np.clip(f, 0.0, 0.999)

        # Update wealth
        W = np.where(
            wins[:, t],
            W * (1.0 + f * model.b),
            W * (1.0 - f),
        )

        if store_paths:
            paths[:, t + 1] = W[idx]

    return W, paths


def run_experiment(
    model: MarketModel,
    cfg: SimulationConfig,
    strategies: Iterable[Strategy],
    *,
    ci_bootstrap_samples: int = 800,
    ci_level: float = 0.95,
) -> Tuple[Dict[str, MetricsResult], Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]]]:
    """
    Run simulation for multiple strategies.

    Returns:
      metrics_by_name: dict name -> MetricsResult
      terminal_by_name: dict name -> terminal wealth samples
      paths_by_name: dict name -> sample paths (or None)
    """
    metrics_by_name: Dict[str, MetricsResult] = {}
    terminal_by_name: Dict[str, np.ndarray] = {}
    paths_by_name: Dict[str, Optional[np.ndarray]] = {}

    # Use different seeds per strategy for independent comparisons with same base seed.
    base_seed = cfg.seed if cfg.seed is not None else None

    for i, strat in enumerate(strategies):
        strat_seed = None if base_seed is None else int(base_seed + 1000 * i)
        cfg_i = SimulationConfig(
            rounds=cfg.rounds,
            trials=cfg.trials,
            initial_wealth=cfg.initial_wealth,
            ruin_threshold=cfg.ruin_threshold,
            seed=strat_seed,
        )
        terminal, paths = simulate_strategy(model, cfg_i, strat, return_paths=True)
        terminal_by_name[strat.name] = terminal
        paths_by_name[strat.name] = paths

        metrics = compute_metrics(
            terminal_wealth=terminal,
            sample_paths=paths,
            initial_wealth=cfg.initial_wealth,
            ruin_threshold=cfg.ruin_threshold,
            ci_bootstrap_samples=ci_bootstrap_samples,
            ci_level=ci_level,
            seed=strat_seed,
        )
        metrics_by_name[strat.name] = metrics

    return metrics_by_name, terminal_by_name, paths_by_name
