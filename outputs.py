"""FIBBar: the output record emitted when a bar closes."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd


@dataclass
class FIBBar:
    """
    A completed Fisher Information Bar.

    Temporal
    --------
    open_time, close_time : float
        Unix epoch seconds.
    duration_seconds : float
        close_time - open_time.
    n_events : int
        Number of market events contained in this bar.

    Price / volume
    --------------
    open, high, low, close : float
        Standard OHLC from the price field.
    sum_volume : float
        Sum of the size field over all events.
    dollar_value : float
        Sum of price * size over all events.
    mean_spread : float | None
        Mean bid-ask spread, or None if quotes were unavailable.

    Information geometry
    --------------------
    information_scalar : float
        Φ(I_k) — the scalarised accumulated information at bar close.
    threshold_at_close : float
        I* — the Information Quantum in effect when the bar closed.
    timeout_flag : bool
        True if the bar was force-closed by a timeout rule rather than
        by the information threshold being reached.

    Provenance
    ----------
    model_name : str
    info_mode : str
    scalarizer_name : str
    start_event_index, end_event_index : int
        Indices into the original event stream.
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

    # Provenance
    model_name: str
    info_mode: str
    scalarizer_name: str
    start_event_index: int
    end_event_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def to_dataframe(cls, bars: list["FIBBar"]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame([b.to_dict() for b in bars])
