"""
fibars — Fisher Information Bars
=================================
An information-geometric approach to financial sampling.

A bar closes not when a clock ticks or a volume bucket fills, but when the
current unfinished bar has accumulated a target amount of statistical
information about the local market process.

Public API
----------
    from fibars import build_fib_bars, FIBConfig, StreamingBuilder
    from fibars.events import MarketEvent
    from fibars import build_time_bars, build_tick_bars, build_volume_bars, build_dollar_bars
"""
from __future__ import annotations

from .config import FIBConfig
from .events import MarketEvent
from .bars.outputs import FIBBar
from .bars.fib_builder import FIBBuilder
from .api.batch import (
    build_fib_bars,
    build_time_bars,
    build_tick_bars,
    build_volume_bars,
    build_dollar_bars,
)
from .api.streaming import StreamingBuilder

__all__ = [
    "FIBConfig",
    "MarketEvent",
    "FIBBar",
    "FIBBuilder",
    "build_fib_bars",
    "build_time_bars",
    "build_tick_bars",
    "build_volume_bars",
    "build_dollar_bars",
    "StreamingBuilder",
]

__version__ = "1.0.0"
