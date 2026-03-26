"""Hawkes process model: theta = (mu, alpha, beta)."""
from __future__ import annotations
import numpy as np
from ..events import MarketEvent
from .base import BaseLocalModel


class HawkesModel(BaseLocalModel):
    """
    Exponential-kernel Hawkes process.

    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

    The recursive excitation state R satisfies:
        R(t) = exp(-beta * dt) * (R(t_prev) + 1)

    Score is the gradient of the log conditional intensity log-likelihood.
    """

    def __init__(
        self,
        mu_init: float = 0.1,
        alpha_init: float = 0.5,
        beta_init: float = 1.0,
        intensity_floor: float = 1e-8,
    ) -> None:
        self._mu_init = max(mu_init, 1e-10)
        self._alpha_init = max(alpha_init, 0.0)
        self._beta_init = max(beta_init, 1e-10)
        self._intensity_floor = max(intensity_floor, 1e-300)

        self._mu: float = self._mu_init
        self._alpha: float = self._alpha_init
        self._beta: float = self._beta_init
        self._R: float = 0.0
        self._prev_time: float | None = None
        self._n: int = 0

    @property
    def n_params(self) -> int:
        return 3

    def initialize(self) -> None:
        self._mu = self._mu_init
        self._alpha = self._alpha_init
        self._beta = self._beta_init
        self._R = 0.0
        self._prev_time = None
        self._n = 0

    def _intensity(self, R: float) -> float:
        """lambda = mu + alpha * R, floored."""
        return max(self._mu + self._alpha * R, self._intensity_floor)

    def _decay(self, dt: float) -> float:
        """Decay factor exp(-beta * dt), clipped to avoid underflow."""
        return float(np.exp(-self._beta * max(dt, 0.0)))

    def update(self, event: MarketEvent) -> None:
        if self._prev_time is not None:
            dt = event.timestamp - self._prev_time
            decay = self._decay(dt)
            self._R = decay * (self._R + 1.0)
        self._prev_time = event.timestamp
        self._n += 1

    def score(self, event: MarketEvent) -> np.ndarray:
        """Score at current event given prior state."""
        if self._prev_time is None:
            return np.zeros(3)
        dt = event.timestamp - self._prev_time
        decay = self._decay(dt)
        R_pre_decay = self._R  # excitation before this event's decay
        R_post_decay = decay * (R_pre_decay + 1.0)

        lam = self._intensity(R_post_decay)

        # d lambda / d theta
        d_lam_d_mu = 1.0
        d_lam_d_alpha = R_post_decay
        d_lam_d_beta = -self._alpha * R_post_decay * dt

        # Score contribution: d/d_theta [log lambda(t_i) - integral lambda dt]
        # For the arrival term: d log lambda / d theta
        # We use the simplified per-event gradient (no compensator derivative here —
        # compensator is implicitly handled by normalization across bar)
        inv_lam = 1.0 / lam
        s_mu = inv_lam * d_lam_d_mu
        s_alpha = inv_lam * d_lam_d_alpha
        s_beta = inv_lam * d_lam_d_beta

        score = np.array([s_mu, s_alpha, s_beta])
        if not np.all(np.isfinite(score)):
            return np.zeros(3)
        return score

    def _log_lik_contrib(
        self, lam: float, mu: float, alpha: float, dt: float, R_pre_decay: float
    ) -> float:
        """Log-lik contribution for diagnostics/testing."""
        decay = self._decay(dt)
        R_new = decay * (R_pre_decay + 1.0)
        lam_new = self._intensity(R_new)
        return float(np.log(max(lam_new, self._intensity_floor)))

    def state_dict(self) -> dict:
        return {
            "mu": self._mu,
            "alpha": self._alpha,
            "beta": self._beta,
            "R": self._R,
            "n": self._n,
        }
