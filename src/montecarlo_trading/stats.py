from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval (percentile method).
    Returns (low, high).
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1)")

    rng = np.random.default_rng(seed)
    x = np.asarray(data)
    n = x.shape[0]
    stats = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(stat_fn(x[idx]))

    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    return low, high
