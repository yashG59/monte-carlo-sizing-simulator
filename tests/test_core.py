import numpy as np

from montecarlo_trading.engine import MarketModel, SimulationConfig, simulate_strategy
from montecarlo_trading.strategies import FixedFraction, kelly_fraction


def test_kelly_fraction_basic():
    # Even odds, p=0.5 => zero edge => f*=0
    assert abs(kelly_fraction(0.5, 1.0) - 0.0) < 1e-12


def test_simulation_shapes():
    model = MarketModel(p=0.55, b=1.0)
    cfg = SimulationConfig(rounds=10, trials=1000, seed=1)
    strat = FixedFraction(0.02)
    terminal, paths = simulate_strategy(model, cfg, strat, return_paths=True, sample_paths=25)
    assert terminal.shape == (1000,)
    assert paths is not None
    assert paths.shape == (25, 11)


def test_wealth_positive_with_small_f():
    model = MarketModel(p=0.55, b=1.0)
    cfg = SimulationConfig(rounds=50, trials=2000, seed=2)
    strat = FixedFraction(0.05)
    terminal, _ = simulate_strategy(model, cfg, strat)
    assert np.all(terminal > 0)
