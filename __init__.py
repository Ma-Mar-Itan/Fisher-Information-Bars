"""
fibars — Fisher Information Bars
=================================
Public API
----------
    from fibars import build_fib_bars, build_fib_bars_from_events
    from fibars import augment_with_fib_features, build_baseline_bars
    from fibars import FIBConfig, StreamingBuilder
"""
from __future__ import annotations

from .config import FIBConfig
from .events import MarketEvent
from .bars.outputs import FIBBar
from .bars.fib_builder import FIBBuilder
from .api.batch import (
    build_fib_bars,
    build_fib_bars_from_events,
    augment_with_fib_features,
    build_baseline_bars,
    build_time_bars,
    build_tick_bars,
    build_volume_bars,
    build_dollar_bars,
)
from .api.streaming import StreamingBuilder

__all__ = [
    "FIBConfig", "MarketEvent", "FIBBar", "FIBBuilder",
    "build_fib_bars", "build_fib_bars_from_events",
    "augment_with_fib_features", "build_baseline_bars",
    "build_time_bars", "build_tick_bars", "build_volume_bars", "build_dollar_bars",
    "StreamingBuilder",
]

__version__ = "1.1.0"
