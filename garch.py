"""GARCH(1,1) quasi-likelihood model.

Parameterization: theta = [log_omega, logit_alpha, logit_beta] with
persistence constraint alpha+beta <= persistence_max.

Score via automatic differentiation approximation using finite differences
of the quasi log-likelihood. This is pragmatic but numerically stable.

QL contribution per observation (given sigma_t^2):
  ell_t = -0.5 * (log(2*pi) + log(sigma_t^2) + eps_t^2 / sigma_t^2)
"""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import BaseLocalModel
from ..events import MarketEvent
from ..types import FloatArray, Matrix

_EPS = 1e-8


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _logit(p: float, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> float:
    p = np.clip(p, lo, hi)
    return float(np.log(p / (1 - p)))


class GARCHModel(BaseLocalModel):
    name = "garch"

    def __init__(
        self,
        omega_init: float = 1e-6,
        alpha_init: float = 0.10,
        beta_init: float = 0.85,
        persistence_max: float = 0.9999,
        var_floor: float = 1e-12,
    ) -> None:
        self._omega = omega_init
        self._alpha = alpha_init
        self._beta = beta_init
        self._persistence_max = persistence_max
        self._var_floor = var_floor
        self._sigma2: float = omega_init / max(1.0 - alpha_init - beta_init, 1e-4)
        self._prev_price: float | None = None
        self._prev_eps2: float = 0.0
        self._n: int = 0

    def initialize(self, **kwargs: Any) -> None:
        self._omega = float(kwargs.get("omega", self._omega))
        self._alpha = float(kwargs.get("alpha", self._alpha))
        self._beta = float(kwargs.get("beta", self._beta))
        # Enforce persistence cap on stored params
        ab = self._alpha + self._beta
        if ab >= self._persistence_max:
            scale = (self._persistence_max - 1e-6) / ab
            self._alpha *= scale
            self._beta *= scale
        unconditional = self._omega / max(1 - self._alpha - self._beta, 1e-4)
        self._sigma2 = max(unconditional, self._var_floor)
        self._prev_price = None
        self._prev_eps2 = 0.0
        self._n = 0

    def _return(self, event: MarketEvent) -> float | None:
        if self._prev_price is None:
            return None
        return event.price - self._prev_price

    def _update_sigma2(self, eps2: float) -> None:
        alpha_b = self._alpha + self._beta
        if alpha_b >= self._persistence_max:
            scale = (self._persistence_max - 1e-6) / alpha_b
            a = self._alpha * scale
            b = self._beta * scale
        else:
            a, b = self._alpha, self._beta
        new_sigma2 = self._omega + a * eps2 + b * self._sigma2
        self._sigma2 = max(new_sigma2, self._var_floor)

    def update(self, event: MarketEvent) -> None:
        r = self._return(event)
        self._prev_price = event.price
        if r is None:
            return
        eps2 = r * r
        self._update_sigma2(eps2)
        self._prev_eps2 = eps2
        self._n += 1

    def _ql(self, r: float, sigma2: float) -> float:
        s2 = max(sigma2, self._var_floor)
        return -0.5 * (np.log(s2) + r * r / s2)

    def score(self, event: MarketEvent) -> FloatArray:
        r = self._return(event)
        if r is None:
            return np.zeros(3)
        # Numerical score via finite differences on [log_omega, logit_alpha, logit_beta]
        params = np.array([
            np.log(max(self._omega, _EPS)),
            _logit(self._alpha),
            _logit(self._beta),
        ])
        h = 1e-5
        grad = np.zeros(3)
        base_ll = self._ql(r, self._sigma2)
        for i in range(3):
            dp = params.copy(); dp[i] += h
            omega_p = np.exp(dp[0])
            alpha_p = _sigmoid(dp[1])
            beta_p = _sigmoid(dp[2])
            if alpha_p + beta_p >= self._persistence_max:
                grad[i] = 0.0
                continue
            sigma2_p = max(omega_p + alpha_p * self._prev_eps2 + beta_p * self._sigma2,
                           self._var_floor)
            grad[i] = (self._ql(r, sigma2_p) - base_ll) / h
        return grad

    def expected_information_increment(self, event: MarketEvent) -> Matrix:
        # Plug-in approximation: outer product of score evaluated at current state
        s = self.score(event)
        return np.outer(s, s)

    def state_dict(self) -> dict[str, Any]:
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "sigma2": self._sigma2,
            "n": self._n,
        }
