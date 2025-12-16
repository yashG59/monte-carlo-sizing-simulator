from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .stats import bootstrap_ci


@dataclass(frozen=True)
class MetricsResult:
    n: int
    terminal_mean: float
    terminal_median: float
    terminal_p10: float
    terminal_p90: float
    loss_probability: float
    ruin_probability: float
    mean_log_growth: float
    vol_log_return: float
    max_drawdown_mean: Optional[float]
    ci: Dict[str, Tuple[float, float]]


def _max_drawdown(path: np.ndarray) -> float:
    # path shape (T,)
    peak = np.maximum.accumulate(path)
    dd = 1.0 - (path / peak)
    return float(np.max(dd))


def compute_metrics(
    *,
    terminal_wealth: np.ndarray,
    sample_paths: Optional[np.ndarray],
    initial_wealth: float,
    ruin_threshold: float,
    ci_bootstrap_samples: int,
    ci_level: float,
    seed: Optional[int],
) -> MetricsResult:
    W = terminal_wealth.astype(np.float64)
    n = int(W.shape[0])

    terminal_mean = float(np.mean(W))
    terminal_median = float(np.median(W))
    terminal_p10 = float(np.percentile(W, 10))
    terminal_p90 = float(np.percentile(W, 90))
    loss_probability = float(np.mean(W < initial_wealth))
    ruin_line = float(initial_wealth * ruin_threshold)
    ruin_probability = float(np.mean(W < ruin_line))

    # log growth over horizon: log(W_T / W_0)
    log_growth = np.log(W / float(initial_wealth))
    mean_log_growth = float(np.mean(log_growth))

    # proxy for per-round volatility: use log returns approximated by total log growth / rounds is not available here,
    # but we can compute dispersion in total log growth across trials.
    vol_log_return = float(np.std(log_growth, ddof=1))

    max_dd_mean = None
    if sample_paths is not None:
        mdds = np.array([_max_drawdown(sample_paths[i]) for i in range(sample_paths.shape[0])], dtype=np.float64)
        max_dd_mean = float(np.mean(mdds))

    # Bootstrap CIs for key metrics
    ci = {}
    ci["terminal_mean"] = bootstrap_ci(W, np.mean, n_boot=ci_bootstrap_samples, level=ci_level, seed=seed)
    ci["loss_probability"] = bootstrap_ci(W, lambda x: np.mean(x < initial_wealth), n_boot=ci_bootstrap_samples, level=ci_level, seed=seed)
    ci["ruin_probability"] = bootstrap_ci(W, lambda x: np.mean(x < ruin_line), n_boot=ci_bootstrap_samples, level=ci_level, seed=seed)
    ci["mean_log_growth"] = bootstrap_ci(log_growth, np.mean, n_boot=ci_bootstrap_samples, level=ci_level, seed=seed)

    return MetricsResult(
        n=n,
        terminal_mean=terminal_mean,
        terminal_median=terminal_median,
        terminal_p10=terminal_p10,
        terminal_p90=terminal_p90,
        loss_probability=loss_probability,
        ruin_probability=ruin_probability,
        mean_log_growth=mean_log_growth,
        vol_log_return=vol_log_return,
        max_drawdown_mean=max_dd_mean,
        ci=ci,
    )
