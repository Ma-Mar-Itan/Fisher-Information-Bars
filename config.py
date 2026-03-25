"""FIBConfig: single source of truth for all hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FIBConfig:
    """
    All knobs for the FIB engine in one place.

    Model selection
    ---------------
    model : "gaussian" | "garch" | "hawkes"
        Local likelihood model used to compute scores.
    info_mode : "observed" | "expected"
        Whether to use the OPG (observed) or analytic (expected)
        Fisher information increment at each event.

    Scalarization
    -------------
    scalarizer : "logdet" | "trace" | "frobenius"
        How to collapse the accumulated information matrix to a scalar.
        "logdet" is reparameterisation-invariant and the recommended default.
    eps_ridge : float
        Ridge regularisation added to J before scalarization.  Prevents
        log-det from diverging to -inf on singular matrices.

    Threshold / Information Quantum
    --------------------------------
    eta : float
        Dimensionless multiplier on the adaptive threshold.
        Higher eta → larger bars (more information per bar).
    delta0_seconds : float
        Reference bar duration in seconds used to seed the adaptive rate.
    ewma_alpha : float
        EWMA smoothing coefficient for the long-run information rate λ̄_Φ.
        Smaller → slower adaptation (more stable threshold).
    min_threshold : float | None
        Hard floor on I* to prevent runaway threshold collapse.
    max_threshold : float | None
        Hard ceiling on I* to prevent bar starvation in dead markets.
    min_warmup_events : int
        Number of *bars* to complete before EWMA is considered reliable.
        During warmup the threshold seeds from the last observed scalar.

    Timeout / safety valves
    -----------------------
    timeout_seconds : float
        Max wall-clock seconds a bar may stay open.
    max_events_per_bar : int
        Hard cap on events per bar.
    inactivity_timeout_seconds : float | None
        If set, close the bar if no event arrives for this many seconds.

    Price / volume fields
    ---------------------
    price_field : str
        Name of the price field on MarketEvent used for OHLC.
    volume_field : str
        Name of the size field on MarketEvent used for volume.

    Model-specific
    --------------
    var_floor : float
        Gaussian variance floor (prevents division by zero).
    garch_persistence_max : float
        Cap on alpha+beta for GARCH stationarity.
    hawkes_intensity_floor : float
        Floor on lambda(t) for Hawkes numerical stability.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    model: Literal["gaussian", "garch", "hawkes"] = "gaussian"
    info_mode: Literal["observed", "expected"] = "observed"

    # ── Scalarization ──────────────────────────────────────────────────────
    scalarizer: Literal["logdet", "trace", "frobenius"] = "logdet"
    eps_ridge: float = 1e-6

    # ── Threshold ──────────────────────────────────────────────────────────
    eta: float = 1.0
    delta0_seconds: float = 60.0
    ewma_alpha: float = 0.05
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    min_warmup_events: int = 20

    # ── Timeout ────────────────────────────────────────────────────────────
    timeout_seconds: float = 300.0
    max_events_per_bar: int = 10_000
    inactivity_timeout_seconds: Optional[float] = None

    # ── Field names ────────────────────────────────────────────────────────
    price_field: str = "price"
    volume_field: str = "size"

    # ── Model-specific ─────────────────────────────────────────────────────
    var_floor: float = 1e-12
    garch_persistence_max: float = 0.9999
    hawkes_intensity_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if self.delta0_seconds <= 0:
            raise ValueError(f"delta0_seconds must be positive, got {self.delta0_seconds}")
        if not (0 < self.ewma_alpha <= 1):
            raise ValueError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")
        if self.eps_ridge < 0:
            raise ValueError(f"eps_ridge must be non-negative, got {self.eps_ridge}")
        if self.model not in ("gaussian", "garch", "hawkes"):
            raise ValueError(f"Unknown model '{self.model}'")
        if self.info_mode not in ("observed", "expected"):
            raise ValueError(f"info_mode must be 'observed' or 'expected'")
        if self.scalarizer not in ("logdet", "trace", "frobenius"):
            raise ValueError(f"Unknown scalarizer '{self.scalarizer}'")
