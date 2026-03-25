"""Univariate Hawkes process with exponential kernel.

Intensity: lambda(t) = mu + alpha * R(t)
Recursive excitation: R(t) = sum_{t_i < t} exp(-beta*(t - t_i))
                    = exp(-beta*dt) * (R_prev + 1)  on each event.

Log-likelihood contribution of event at t_i:
  ell_i = log(lambda(t_i)) - integral_{t_{i-1}}^{t_i} lambda(s) ds
        = log(mu + alpha*R_i) - mu*dt - (alpha/beta)*(R_prev - R_i + 1)

Score w.r.t. theta = [log_mu, log_alpha, log_beta] via finite differences.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import BaseLocalModel
from ..events import MarketEvent
from ..types import FloatArray, Matrix

_EPS = 1e-8


class HawkesModel(BaseLocalModel):
    name = "hawkes"

    def __init__(
        self,
        mu_init: float = 1.0,
        alpha_init: float = 0.5,
        beta_init: float = 1.0,
        intensity_floor: float = 1e-8,
    ) -> None:
        self._mu = mu_init
        self._alpha = alpha_init
        self._beta = beta_init
        self._floor = intensity_floor
        self._R: float = 0.0          # recursive excitation state
        self._prev_time: float | None = None
        self._n: int = 0

    def initialize(self, **kwargs: Any) -> None:
        self._mu = float(kwargs.get("mu", self._mu))
        self._alpha = float(kwargs.get("alpha", self._alpha))
        self._beta = float(kwargs.get("beta", self._beta))
        self._R = 0.0
        self._prev_time = None
        self._n = 0

    def _dt(self, event: MarketEvent) -> float:
        if self._prev_time is None:
            return 0.0
        return max(event.timestamp - self._prev_time, 0.0)

    def _decay_R(self, dt: float) -> float:
        return self._R * np.exp(-self._beta * dt)

    def _intensity(self, R: float) -> float:
        return max(self._mu + self._alpha * R, self._floor)

    def update(self, event: MarketEvent) -> None:
        dt = self._dt(event)
        self._R = self._decay_R(dt) + 1.0   # +1 at arrival
        self._prev_time = event.timestamp
        self._n += 1

    def _log_lik_contrib(
        self,
        mu: float, alpha: float, beta: float,
        dt: float, R_pre_decay: float,
    ) -> float:
        """LL contribution for one event."""
        R_decayed = R_pre_decay * np.exp(-beta * dt)
        R_post = R_decayed + 1.0
        lam = max(mu + alpha * R_post, self._floor)
        # Integral term: mu*dt + (alpha/beta)*(R_pre_decay - R_decayed)
        integral = mu * dt + (alpha / max(beta, _EPS)) * (R_pre_decay - R_decayed)
        return np.log(lam) - integral

    def score(self, event: MarketEvent) -> FloatArray:
        dt = self._dt(event)
        R_pre = self._decay_R(dt)  # before adding +1 for this event
        params = np.array([
            np.log(max(self._mu, _EPS)),
            np.log(max(self._alpha, _EPS)),
            np.log(max(self._beta, _EPS)),
        ])
        h = 1e-5
        base_ll = self._log_lik_contrib(self._mu, self._alpha, self._beta, dt, R_pre)
        grad = np.zeros(3)
        for i in range(3):
            dp = params.copy(); dp[i] += h
            mu_p = np.exp(dp[0])
            alpha_p = np.exp(dp[1])
            beta_p = np.exp(dp[2])
            ll_p = self._log_lik_contrib(mu_p, alpha_p, beta_p, dt, R_pre)
            grad[i] = (ll_p - base_ll) / h
        return grad

    def expected_information_increment(self, event: MarketEvent) -> Matrix:
        s = self.score(event)
        return np.outer(s, s)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mu": self._mu,
            "alpha": self._alpha,
            "beta": self._beta,
            "R": self._R,
            "n": self._n,
        }
