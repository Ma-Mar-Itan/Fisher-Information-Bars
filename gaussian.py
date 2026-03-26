"""Local Gaussian model: theta = (mu, sigma^2)."""
from __future__ import annotations
import numpy as np
from ..events import MarketEvent
from .base import BaseLocalModel


class GaussianModel(BaseLocalModel):
    """
    Online Welford estimator for N(mu, sigma^2).

    Score:
      d/d_mu      log f = (x - mu) / sigma^2
      d/d_sigma^2 log f = -1/(2*sigma^2) + (x-mu)^2 / (2*sigma^4)

    Expected Fisher (diagonal):
      I_11 = 1 / sigma^2
      I_22 = 1 / (2 * sigma^4)
    """

    def __init__(self, var_floor: float = 1e-12) -> None:
        self._var_floor = max(var_floor, 1e-300)
        self._mu: float = 0.0
        self._sigma2: float = 1.0
        self._n: int = 0
        self._M2: float = 0.0  # Welford accumulator

    @property
    def n_params(self) -> int:
        return 2

    def initialize(self) -> None:
        self._mu = 0.0
        self._sigma2 = 1.0
        self._n = 0
        self._M2 = 0.0

    def update(self, event: MarketEvent) -> None:
        x = event.price
        self._n += 1
        delta = x - self._mu
        self._mu += delta / self._n
        delta2 = x - self._mu
        self._M2 += delta * delta2
        if self._n >= 2:
            self._sigma2 = max(self._M2 / (self._n - 1), self._var_floor)

    def score(self, event: MarketEvent) -> np.ndarray:
        if self._n == 0:
            return np.zeros(2)
        x = event.price
        s2 = max(self._sigma2, self._var_floor)
        resid = x - self._mu
        s_mu = resid / s2
        s_s2 = -0.5 / s2 + 0.5 * resid**2 / s2**2
        return np.array([s_mu, s_s2])

    def expected_information_increment(self, event: MarketEvent) -> np.ndarray:
        if self._n == 0:
            return np.zeros((2, 2))
        s2 = max(self._sigma2, self._var_floor)
        I = np.zeros((2, 2))
        I[0, 0] = 1.0 / s2
        I[1, 1] = 0.5 / (s2 ** 2)
        return I

    def state_dict(self) -> dict:
        return {"mu": self._mu, "sigma2": self._sigma2, "n": self._n}
