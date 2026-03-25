"""Adaptive Information Quantum policy.

I* = eta * lambda_bar_Phi * delta0

where lambda_bar_Phi is an EWMA of the scalarised information rate
(scalar / duration_seconds) across completed bars.

During warmup (fewer than min_warmup_events completed bars), the threshold
seeds from the first observed scalar so the engine is never deadlocked at inf.
"""
from __future__ import annotations
from typing import Optional


class AdaptiveThresholdPolicy:
    """
    Tracks the asset's long-run information rate and emits I* on demand.

    Parameters
    ----------
    eta : float
        Multiplier on the adaptive threshold.
    delta0_seconds : float
        Reference bar duration.  The threshold targets bars that would take
        approximately delta0_seconds seconds at the current information rate.
    ewma_alpha : float
        EWMA smoothing coefficient.  Smaller = more stable / slower adaptation.
    min_threshold : float | None
        Hard floor applied after the formula.
    max_threshold : float | None
        Hard ceiling applied after the formula.
    min_warmup_events : int
        Number of completed bars before EWMA is considered reliable.
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

        # State
        self._n_bars_completed: int = 0
        self._ewma_rate: Optional[float] = None   # lambda_bar_Phi (scalar / second)
        self._last_scalar: float = 0.0            # fallback during warmup
        self._threshold: float = float("inf")     # current I*
        self._seeded: bool = False                # True once first scalar observed

    # ── Public interface ────────────────────────────────────────────────────

    def current_threshold(self) -> float:
        """Return the current Information Quantum I*."""
        return self._threshold

    def seed_from_scalar(self, scalar: float) -> None:
        """
        Called by FIBBuilder with the first non-zero scalar it computes.

        Without this, _threshold stays at inf until the first bar closes,
        but no bar can close until _threshold drops — a deadlock.
        Seeding from the first observed scalar breaks that cycle.
        """
        if not self._seeded and scalar > 0.0:
            self._last_scalar = scalar
            raw = scalar  # seed at current level; eta/EWMA refine after warmup
            if self._min is not None:
                raw = max(raw, self._min)
            if self._max is not None:
                raw = min(raw, self._max)
            self._threshold = raw
            self._seeded = True

    def update(self, bar_scalar: float, bar_duration_seconds: float) -> None:
        """
        Call once per completed bar with its final scalar and wall-clock duration.
        Updates the EWMA rate and recomputes I*.
        """
        self._last_scalar = bar_scalar
        self._n_bars_completed += 1

        # Instantaneous information rate for this bar
        dur = max(bar_duration_seconds, 1e-3)
        rate = bar_scalar / dur

        # Update EWMA
        if self._ewma_rate is None:
            self._ewma_rate = rate
        else:
            self._ewma_rate = self._alpha * rate + (1.0 - self._alpha) * self._ewma_rate

        self._recompute()

    # ── Internal ────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        """Recompute I* from current EWMA rate."""
        if self._n_bars_completed < self._warmup or self._ewma_rate is None:
            # Still in warmup: use last observed scalar as a live seed
            raw = max(self._last_scalar, 1e-9)
        else:
            raw = self._eta * self._ewma_rate * self._delta0

        if self._min is not None:
            raw = max(raw, self._min)
        if self._max is not None:
            raw = min(raw, self._max)

        self._threshold = raw

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
