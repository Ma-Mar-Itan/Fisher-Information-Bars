"""Adaptive Information Quantum policy.

I* = eta * lambda_bar_Phi * delta0

where lambda_bar_Phi is an EWMA of the scalarised information rate
(scalar / duration_seconds) across completed bars.
"""
from __future__ import annotations
from typing import Optional


class AdaptiveThresholdPolicy:
    """
    Tracks the asset's long-run information rate and emits I* on demand.

    Parameters
    ----------
    eta : float             — multiplier
    delta0_seconds : float  — reference bar duration
    ewma_alpha : float      — EWMA smoothing (smaller = slower)
    min_threshold : float   — hard floor
    max_threshold : float   — hard ceiling
    min_warmup_events : int — bars before EWMA is trusted
    """

    def __init__(
        self,
        eta: float = 1.0,
        delta0_seconds: float = 60.0,
        ewma_alpha: float = 0.05,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        min_warmup_events: int = 20,
    ) -> None:
        self._eta = eta
        self._delta0 = delta0_seconds
        self._alpha = ewma_alpha
        self._min = min_threshold
        self._max = max_threshold
        self._warmup = min_warmup_events

        self._n_bars_completed: int = 0
        self._ewma_rate: Optional[float] = None
        self._last_scalar: float = 0.0
        self._threshold: float = float("inf")
        self._seeded: bool = False

    def current_threshold(self) -> float:
        return self._threshold

    def seed_from_scalar(self, scalar: float) -> None:
        """
        Called with the first non-zero scalar to break the inf-deadlock.
        Only fires once.
        """
        if not self._seeded and scalar > 0.0:
            self._last_scalar = scalar
            raw = scalar
            self._threshold = self._apply_bounds(raw)
            self._seeded = True

    def update(self, bar_scalar: float, bar_duration_seconds: float) -> None:
        """Call once per completed bar."""
        self._last_scalar = bar_scalar
        self._n_bars_completed += 1
        dur = max(bar_duration_seconds, 1e-3)
        rate = bar_scalar / dur
        if self._ewma_rate is None:
            self._ewma_rate = rate
        else:
            self._ewma_rate = self._alpha * rate + (1.0 - self._alpha) * self._ewma_rate
        self._recompute()

    def _recompute(self) -> None:
        if self._n_bars_completed < self._warmup or self._ewma_rate is None:
            raw = max(self._last_scalar, 1e-9)
        else:
            raw = self._eta * self._ewma_rate * self._delta0
        self._threshold = self._apply_bounds(raw)

    def _apply_bounds(self, raw: float) -> float:
        if self._min is not None:
            raw = max(raw, self._min)
        if self._max is not None:
            raw = min(raw, self._max)
        return raw

    @property
    def n_bars_completed(self) -> int:
        return self._n_bars_completed

    @property
    def ewma_rate(self) -> Optional[float]:
        return self._ewma_rate

    def state_dict(self) -> dict:
        return {
            "n_bars_completed": self._n_bars_completed,
            "ewma_rate": self._ewma_rate,
            "last_scalar": self._last_scalar,
            "threshold": self._threshold,
            "seeded": self._seeded,
        }
