"""MarketEvent: the atomic input unit fed to the FIB engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketEvent:
    """
    One market message (tick, trade, or quote update).

    Required
    --------
    timestamp : float
        Unix epoch seconds (float for sub-second precision).
    price : float
        Last trade price or mid-quote.

    Optional
    --------
    size : float
        Trade size / number of contracts.
    bid : float | None
        Best bid price at event time.
    ask : float | None
        Best ask price at event time.
    side : str | None
        'B' (buyer-initiated) | 'S' (seller-initiated) | None.
    event_type : str | None
        e.g. 'trade', 'quote', 'cancel'. Reserved for future model stages.
    index : int
        Global sequential index; set by FIBBuilder, not the caller.
    """
    timestamp: float
    price: float
    size: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    side: Optional[str] = None
    event_type: Optional[str] = None
    index: int = field(default=0, repr=False)

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread, or None if quotes unavailable."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid(self) -> float:
        """Mid-quote if both sides present, else price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.price
