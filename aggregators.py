"""BarAggregator: tracks OHLCV and timing for the open bar."""
from __future__ import annotations
from typing import Optional
from ..events import MarketEvent


class BarAggregator:
    """
    Accumulates raw OHLCV statistics for the events in the current bar.

    Tracks:
      - open / high / low / close prices
      - sum of volume (size)
      - dollar value (sum of price * size)
      - mean bid-ask spread (only when bid/ask are present)
      - open_time / close_time (timestamps of first/last event)
      - start_index / end_index (global event indices)
    """

    def __init__(
        self,
        price_field: str = "price",
        volume_field: str = "size",
    ) -> None:
        self._price_field = price_field
        self._volume_field = volume_field
        self.reset()

    def reset(self) -> None:
        self._n: int = 0
        self._open: Optional[float] = None
        self._high: float = float("-inf")
        self._low: float = float("inf")
        self._close: Optional[float] = None
        self._sum_vol: float = 0.0
        self._dollar: float = 0.0
        self._spread_sum: float = 0.0
        self._spread_count: int = 0
        self._open_time: Optional[float] = None
        self._close_time: Optional[float] = None
        self._start_index: int = 0
        self._end_index: int = 0

    def add(self, event: MarketEvent) -> None:
        price = getattr(event, self._price_field, event.price)
        size = getattr(event, self._volume_field, event.size)

        if self._n == 0:
            self._open = price
            self._open_time = event.timestamp
            self._start_index = event.index

        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._close_time = event.timestamp
        self._end_index = event.index
        self._sum_vol += size
        self._dollar += price * size

        spread = event.spread
        if spread is not None and spread >= 0.0:
            self._spread_sum += spread
            self._spread_count += 1

        self._n += 1

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_events(self) -> int:
        return self._n

    @property
    def open(self) -> float:
        return self._open or 0.0

    @property
    def high(self) -> float:
        return self._high if self._n > 0 else 0.0

    @property
    def low(self) -> float:
        return self._low if self._n > 0 else 0.0

    @property
    def close(self) -> float:
        return self._close or 0.0

    @property
    def sum_volume(self) -> float:
        return self._sum_vol

    @property
    def dollar_value(self) -> float:
        return self._dollar

    @property
    def mean_spread(self) -> Optional[float]:
        if self._spread_count == 0:
            return None
        return self._spread_sum / self._spread_count

    @property
    def open_time(self) -> Optional[float]:
        return self._open_time

    @property
    def close_time(self) -> Optional[float]:
        return self._close_time

    @property
    def duration_seconds(self) -> float:
        if self._open_time is None or self._close_time is None:
            return 0.0
        return max(self._close_time - self._open_time, 0.0)

    @property
    def start_index(self) -> int:
        return self._start_index

    @property
    def end_index(self) -> int:
        return self._end_index
