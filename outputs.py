"""FIBBar: the output record emitted when a bar closes."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal
import pandas as pd


CloseReason = Literal["threshold", "timeout", "max_events", "flush", "inactivity"]


@dataclass
class FIBBar:
    """
    A completed Fisher Information Bar.

    Temporal
    --------
    open_time, close_time : float   — Unix epoch seconds
    duration_seconds : float
    n_events : int

    Price / volume
    --------------
    open, high, low, close : float
    sum_volume : float
    dollar_value : float
    mean_spread : float | None

    Information geometry
    --------------------
    information_scalar : float      — Phi(I_k) at bar close
    threshold_at_close : float      — I* at bar close
    timeout_flag : bool             — True if not threshold-triggered
    close_reason : str              — 'threshold' | 'timeout' | 'max_events' | 'flush' | 'inactivity'

    Provenance
    ----------
    model_name, info_mode, scalarizer_name : str
    start_event_index, end_event_index : int
    """

    # Temporal
    open_time: float
    close_time: float
    duration_seconds: float
    n_events: int

    # Price / volume
    open: float
    high: float
    low: float
    close: float
    sum_volume: float
    dollar_value: float
    mean_spread: Optional[float]

    # Information geometry
    information_scalar: float
    threshold_at_close: float
    timeout_flag: bool
    close_reason: str = "threshold"

    # Provenance
    model_name: str = ""
    info_mode: str = ""
    scalarizer_name: str = ""
    start_event_index: int = 0
    end_event_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def to_dataframe(cls, bars: list["FIBBar"]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([b.to_dict() for b in bars])
