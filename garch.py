"""GARCH(1,1) quasi-likelihood model: theta = (omega, alpha, beta)."""
from __future__ import annotations
import numpy as np
from ..events import MarketEvent
from .base import BaseLocalModel


class GARCHModel(BaseLocalModel):
    """
    Online GARCH(1,1) filter with score-driven updates.

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Score is the quasi-score of the Gaussian log-likelihood wrt (omega, alpha, beta).
    Persistence is capped to ensure stationarity.
    """

    def __init__(
        self,
        omega_init: float = 1e-6,
        alpha_init: float = 0.05,
        beta_init: float = 0.90,
        persistence_max: float = 0.9999,
        var_floor: float = 1e-12,
    ) -> None:
        # Clip alpha+beta at init
        total = alpha_init + beta_init
        if total >= persistence_max:
            scale = (persistence_max - 1e-9) / total
            alpha_init *= scale
            beta_init *= scale
        self._omega_init = max(omega_init, 1e-12)
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._persistence_max = persistence_max
        self._var_floor = max(var_floor, 1e-300)

        # Runtime state
        self._omega: float = self._omega_init
        self._alpha: float = self._alpha_init
        self._beta: float = self._beta_init
        self._sigma2: float = self._omega_init / max(1.0 - alpha_init - beta_init, 1e-6)
        self._prev_eps2: float = 0.0
        self._prev_price: float | None = None
        self._n: int = 0

    @property
    def n_params(self) -> int:
        return 3

    def initialize(self) -> None:
        self._sigma2 = self._omega_init / max(1.0 - self._alpha_init - self._beta_init, 1e-6)
        self._prev_eps2 = 0.0
        self._prev_price = None
        self._n = 0
        self._omega = self._omega_init
        self._alpha = self._alpha_init
        self._beta = self._beta_init

    def _update_sigma2(self) -> None:
        """Recompute sigma2 from current params and state."""
        self._sigma2 = max(
            self._omega + self._alpha * self._prev_eps2 + self._beta * self._sigma2,
            self._var_floor,
        )

    def update(self, event: MarketEvent) -> None:
        if self._prev_price is not None:
            eps = event.price - self._prev_price
            self._prev_eps2 = eps * eps
            self._update_sigma2()
        self._prev_price = event.price
        self._n += 1

    def score(self, event: MarketEvent) -> np.ndarray:
        """Quasi-score wrt (omega, alpha, beta) at current state."""
        if self._prev_price is None or self._n == 0:
            return np.zeros(3)
        eps = event.price - self._prev_price
        eps2 = eps * eps
        s2 = max(self._sigma2, self._var_floor)

        # d log L / d sigma2 (scalar intermediate)
        u = eps2 / s2 - 1.0

        # d sigma2 / d theta = (1, eps_{t-1}^2, sigma_{t-1}^2)
        # Chain rule: score_i = (u / (2 * s2)) * d_sigma2_d_theta_i
        factor = u / (2.0 * s2)
        s_omega = factor * 1.0
        s_alpha = factor * self._prev_eps2
        s_beta = factor * self._sigma2
        return np.array([s_omega, s_alpha, s_beta])

    def state_dict(self) -> dict:
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "sigma2": self._sigma2,
            "n": self._n,
        }
