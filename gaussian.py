"""Local Gaussian returns model.

Parameterization: theta = [mu, log_sigma2].
Score w.r.t. [mu, log_sigma2]:
  s_mu        = (r - mu) / sigma2
  s_logsigma2 = -0.5 + 0.5 * (r - mu)^2 / sigma2

Expected information (analytic):
  I_11 = 1/sigma2
  I_22 = 0.5
  off-diagonal = 0
"""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import BaseLocalModel
from ..events import MarketEvent
from ..types import FloatArray, Matrix


class GaussianModel(BaseLocalModel):
    name = "gaussian"

    def __init__(self, var_floor: float = 1e-12) -> None:
        self.var_floor = var_floor
        self._mu: float = 0.0
        self._sigma2: float = 1.0
        self._n: int = 0
        self._prev_price: float | None = None
        # Welford online mean/variance
        self._M2: float = 0.0

    def initialize(self, **kwargs: Any) -> None:
        self._mu = float(kwargs.get("mu", 0.0))
        self._sigma2 = float(kwargs.get("sigma2", 1.0))
        self._n = 0
        self._prev_price = None
        self._M2 = 0.0

    def _return(self, event: MarketEvent) -> float | None:
        if self._prev_price is None:
            return None
        if self._prev_price == 0.0:
            return 0.0
        return event.price - self._prev_price   # arithmetic return

    def update(self, event: MarketEvent) -> None:
        r = self._return(event)
        self._prev_price = event.price
        if r is None:
            return
        # Welford online
        self._n += 1
        delta = r - self._mu
        self._mu += delta / self._n
        delta2 = r - self._mu
        self._M2 += delta * delta2
        if self._n >= 2:
            self._sigma2 = max(self._M2 / (self._n - 1), self.var_floor)

    def score(self, event: MarketEvent) -> FloatArray:
        r = self._return(event)
        if r is None:
            return np.zeros(2)
        sigma2 = max(self._sigma2, self.var_floor)
        resid = r - self._mu
        s_mu = resid / sigma2
        s_log = -0.5 + 0.5 * resid**2 / sigma2
        return np.array([s_mu, s_log])

    def expected_information_increment(self, event: MarketEvent) -> Matrix:
        sigma2 = max(self._sigma2, self.var_floor)
        return np.diag([1.0 / sigma2, 0.5])

    def state_dict(self) -> dict[str, Any]:
        return {"mu": self._mu, "sigma2": self._sigma2, "n": self._n}
