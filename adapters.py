"""DataFrame ↔ MarketEvent adapters."""
from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd
from ..events import MarketEvent
from ..bars.outputs import FIBBar


def df_to_events(df: pd.DataFrame) -> list[MarketEvent]:
    """Convert a DataFrame to a list of MarketEvent objects."""
    _validate_df(df)
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    has_bid = "bid" in df.columns
    has_ask = "ask" in df.columns
    has_size = "size" in df.columns
    has_side = "side" in df.columns
    has_etype = "event_type" in df.columns

    events: list[MarketEvent] = []
    for row in df.itertuples(index=False):
        ev = MarketEvent(
            timestamp=float(row.timestamp),
            price=float(row.price),
            size=float(row.size) if has_size else 0.0,
            bid=float(row.bid) if has_bid and not _isnan(row.bid) else None,
            ask=float(row.ask) if has_ask and not _isnan(row.ask) else None,
            side=str(row.side) if has_side and row.side is not None else None,
            event_type=str(row.event_type) if has_etype and row.event_type is not None else None,
        )
        events.append(ev)
    return events


def df_to_event_stream(df: pd.DataFrame) -> Iterator[MarketEvent]:
    """Lazy generator version for large datasets."""
    _validate_df(df)
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    has_bid = "bid" in df.columns
    has_ask = "ask" in df.columns
    has_size = "size" in df.columns
    has_side = "side" in df.columns
    has_etype = "event_type" in df.columns

    for row in df.itertuples(index=False):
        yield MarketEvent(
            timestamp=float(row.timestamp),
            price=float(row.price),
            size=float(row.size) if has_size else 0.0,
            bid=float(row.bid) if has_bid and not _isnan(row.bid) else None,
            ask=float(row.ask) if has_ask and not _isnan(row.ask) else None,
            side=str(row.side) if has_side and row.side is not None else None,
            event_type=str(row.event_type) if has_etype and row.event_type is not None else None,
        )


def bars_to_df(bars: list[FIBBar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    return FIBBar.to_dataframe(bars)


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in ("timestamp", "price") if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        return
    if df["price"].isnull().any():
        raise ValueError("Column 'price' contains NaN values. Drop or fill before building bars.")
    if df["timestamp"].isnull().any():
        raise ValueError("Column 'timestamp' contains NaN values.")


def _isnan(val) -> bool:
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return val is None
