"""Baseline bar builders (time, tick, volume, dollar) for comparison.

All return the same FIBBar schema with information_scalar=0.0 and
scalarizer_name set to the bar type so they can be compared directly
against FIBs.
"""
from __future__ import annotations
from typing import Iterator
import pandas as pd
from ..events import MarketEvent
from .aggregators import BarAggregator
from .outputs import FIBBar


def _make_bar(agg: BarAggregator, bar_type: str, timeout: bool = False) -> FIBBar:
    return FIBBar(
        open_time=agg.open_time or 0.0,
        close_time=agg.close_time or 0.0,
        duration_seconds=agg.duration_seconds,
        n_events=agg.n_events,
        open=agg.open,
        high=agg.high,
        low=agg.low,
        close=agg.close,
        sum_volume=agg.sum_volume,
        dollar_value=agg.dollar_value,
        mean_spread=agg.mean_spread,
        information_scalar=0.0,
        threshold_at_close=0.0,
        timeout_flag=timeout,
        model_name="baseline",
        info_mode="none",
        scalarizer_name=bar_type,
        start_event_index=agg.start_index,
        end_event_index=agg.end_index,
    )


# ── Time bars ───────────────────────────────────────────────────────────────

def build_time_bars_from_events(
    events: list[MarketEvent],
    seconds_per_bar: float = 60.0,
) -> list[FIBBar]:
    """Close a bar every `seconds_per_bar` wall-clock seconds."""
    bars: list[FIBBar] = []
    agg = BarAggregator()
    bar_end: float | None = None

    for i, ev in enumerate(events):
        ev.index = i + 1
        if agg.n_events == 0:
            agg.add(ev)
            bar_end = ev.timestamp + seconds_per_bar
            continue

        if ev.timestamp >= bar_end:  # type: ignore[operator]
            bars.append(_make_bar(agg, "time"))
            agg.reset()
            bar_end = None

        agg.add(ev)
        if bar_end is None:
            bar_end = ev.timestamp + seconds_per_bar

    if agg.n_events > 0:
        bars.append(_make_bar(agg, "time", timeout=True))
    return bars


# ── Tick bars ────────────────────────────────────────────────────────────────

def build_tick_bars_from_events(
    events: list[MarketEvent],
    ticks_per_bar: int = 100,
) -> list[FIBBar]:
    """Close a bar every `ticks_per_bar` events."""
    bars: list[FIBBar] = []
    agg = BarAggregator()

    for i, ev in enumerate(events):
        ev.index = i + 1
        agg.add(ev)
        if agg.n_events >= ticks_per_bar:
            bars.append(_make_bar(agg, "tick"))
            agg.reset()

    if agg.n_events > 0:
        bars.append(_make_bar(agg, "tick", timeout=True))
    return bars


# ── Volume bars ──────────────────────────────────────────────────────────────

def build_volume_bars_from_events(
    events: list[MarketEvent],
    volume_per_bar: float = 1000.0,
) -> list[FIBBar]:
    """Close a bar when cumulative size crosses `volume_per_bar`."""
    bars: list[FIBBar] = []
    agg = BarAggregator()
    cum_vol: float = 0.0

    for i, ev in enumerate(events):
        ev.index = i + 1
        agg.add(ev)
        cum_vol += ev.size
        if cum_vol >= volume_per_bar:
            bars.append(_make_bar(agg, "volume"))
            agg.reset()
            cum_vol = 0.0

    if agg.n_events > 0:
        bars.append(_make_bar(agg, "volume", timeout=True))
    return bars


# ── Dollar bars ───────────────────────────────────────────────────────────────

def build_dollar_bars_from_events(
    events: list[MarketEvent],
    dollar_per_bar: float = 100_000.0,
) -> list[FIBBar]:
    """Close a bar when cumulative price*size crosses `dollar_per_bar`."""
    bars: list[FIBBar] = []
    agg = BarAggregator()
    cum_dollar: float = 0.0

    for i, ev in enumerate(events):
        ev.index = i + 1
        agg.add(ev)
        cum_dollar += ev.price * ev.size
        if cum_dollar >= dollar_per_bar:
            bars.append(_make_bar(agg, "dollar"))
            agg.reset()
            cum_dollar = 0.0

    if agg.n_events > 0:
        bars.append(_make_bar(agg, "dollar", timeout=True))
    return bars
