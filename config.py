"""FIBConfig: single source of truth for all hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FIBConfig:
    """
    All knobs for the FIB engine in one place.

    Model selection
    ---------------
    model : "gaussian" | "garch" | "hawkes"
    info_mode : "observed" | "expected"

    Scalarization
    -------------
    scalarizer : "logdet" | "trace" | "frobenius"
    eps_ridge : float  — ridge added to J before scalarization

    Threshold / Information Quantum
    --------------------------------
    eta : float          — multiplier on adaptive threshold
    delta0_seconds : float — reference bar duration
    ewma_alpha : float   — EWMA smoothing for long-run rate
    min_threshold : float | None
    max_threshold : float | None
    min_warmup_events : int — bars before EWMA is trusted

    Timeout / safety valves
    -----------------------
    timeout_seconds : float
    max_events_per_bar : int
    inactivity_timeout_seconds : float | None

    Field names
    -----------
    price_field, volume_field : str

    Model-specific
    --------------
    var_floor : float               — Gaussian variance floor
    garch_persistence_max : float   — cap on alpha+beta
    hawkes_intensity_floor : float  — floor on lambda(t)
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
            raise ValueError(f"Unknown model '{self.model}'. Choose: gaussian, garch, hawkes")
        if self.info_mode not in ("observed", "expected"):
            raise ValueError(f"info_mode must be 'observed' or 'expected', got '{self.info_mode}'")
        if self.scalarizer not in ("logdet", "trace", "frobenius"):
            raise ValueError(f"Unknown scalarizer '{self.scalarizer}'. Choose: logdet, trace, frobenius")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.max_events_per_bar < 1:
            raise ValueError(f"max_events_per_bar must be >= 1, got {self.max_events_per_bar}")
        if self.min_warmup_events < 0:
            raise ValueError(f"min_warmup_events must be >= 0, got {self.min_warmup_events}")
        if self.inactivity_timeout_seconds is not None and self.inactivity_timeout_seconds <= 0:
            raise ValueError(f"inactivity_timeout_seconds must be positive, got {self.inactivity_timeout_seconds}")
        if self.min_threshold is not None and self.min_threshold < 0:
            raise ValueError(f"min_threshold must be non-negative")
        if self.max_threshold is not None and self.max_threshold <= 0:
            raise ValueError(f"max_threshold must be positive")
        if self.min_threshold is not None and self.max_threshold is not None:
            if self.min_threshold >= self.max_threshold:
                raise ValueError(f"min_threshold must be < max_threshold")
        if self.var_floor <= 0:
            raise ValueError(f"var_floor must be positive, got {self.var_floor}")
        if not (0 < self.garch_persistence_max < 1):
            raise ValueError(f"garch_persistence_max must be in (0,1), got {self.garch_persistence_max}")
        if self.hawkes_intensity_floor <= 0:
            raise ValueError(f"hawkes_intensity_floor must be positive")
