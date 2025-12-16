from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .engine import MarketModel


class Strategy:
    """
    Base class: implement fraction().

    fraction() may return:
      - scalar float
      - np.ndarray same shape as wealth
    """
    name: str = "Strategy"

    def fraction(self, *, model: MarketModel, wealth: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedFraction(Strategy):
    f: float
    name: str = "FixedFraction"

    def fraction(self, *, model: MarketModel, wealth: np.ndarray, t: int) -> np.ndarray:
        return np.full_like(wealth, self.f, dtype=np.float64)


def kelly_fraction(p: float, b: float) -> float:
    # f* = p - (1-p)/b
    return p - (1.0 - p) / b


@dataclass(frozen=True)
class Kelly(Strategy):
    cap: Optional[float] = None
    name: str = "Kelly"

    def fraction(self, *, model: MarketModel, wealth: np.ndarray, t: int) -> np.ndarray:
        f = float(kelly_fraction(model.p, model.b))
        if self.cap is not None:
            f = min(f, float(self.cap))
        return np.full_like(wealth, f, dtype=np.float64)


@dataclass(frozen=True)
class FractionalKelly(Strategy):
    k: float = 0.5
    cap: Optional[float] = None
    name: str = "FractionalKelly"

    def fraction(self, *, model: MarketModel, wealth: np.ndarray, t: int) -> np.ndarray:
        f = float(self.k) * float(kelly_fraction(model.p, model.b))
        if self.cap is not None:
            f = min(f, float(self.cap))
        return np.full_like(wealth, f, dtype=np.float64)


@dataclass(frozen=True)
class CappedKelly(Strategy):
    cap: float = 0.10
    name: str = "CappedKelly"

    def fraction(self, *, model: MarketModel, wealth: np.ndarray, t: int) -> np.ndarray:
        f = float(kelly_fraction(model.p, model.b))
        f = min(f, float(self.cap))
        return np.full_like(wealth, f, dtype=np.float64)
