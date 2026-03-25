"""Batch API: build bars from a complete DataFrame in one call."""
from __future__ import annotations
from typing import Any
import pandas as pd
from ..config import FIBConfig
from ..bars.fib_builder import FIBBuilder
from ..bars.outputs import FIBBar
from ..bars.baseline import (
    build_time_bars_from_events,
    build_tick_bars_from_events,
    build_volume_bars_from_events,
    build_dollar_bars_from_events,
)
from ..data.adapters import df_to_events, bars_to_df


def build_fib_bars(
    data: pd.DataFrame,
    model: str = "gaussian",
    info_mode: str = "observed",
    scalarizer: str = "logdet",
    eta: float = 1.0,
    delta0_seconds: float = 60.0,
    ewma_alpha: float = 0.05,
    eps_ridge: float = 1e-6,
    timeout_seconds: float = 300.0,
    max_events_per_bar: int = 10_000,
    inactivity_timeout_seconds: float | None = None,
    min_warmup_events: int = 20,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    var_floor: float = 1e-12,
    garch_persistence_max: float = 0.9999,
    hawkes_intensity_floor: float = 1e-8,
    config: FIBConfig | None = None,
) -> pd.DataFrame:
    """
    Build Fisher Information Bars from a tick DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'timestamp' (unix seconds) and 'price'.
        Optional: 'size', 'bid', 'ask', 'side', 'event_type'.
    model : str
        'gaussian' | 'garch' | 'hawkes'
    info_mode : str
        'observed' (OPG) | 'expected' (analytic where available)
    scalarizer : str
        'logdet' | 'trace' | 'frobenius'
    eta : float
        Threshold multiplier.
    delta0_seconds : float
        Reference bar duration for adaptive threshold seeding.
    config : FIBConfig | None
        If provided, all other keyword args are ignored.

    Returns
    -------
    pd.DataFrame
        One row per completed bar.  See FIBBar for column descriptions.
    """
    if config is None:
        config = FIBConfig(
            model=model,
            info_mode=info_mode,
            scalarizer=scalarizer,
            eta=eta,
            delta0_seconds=delta0_seconds,
            ewma_alpha=ewma_alpha,
            eps_ridge=eps_ridge,
            timeout_seconds=timeout_seconds,
            max_events_per_bar=max_events_per_bar,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            min_warmup_events=min_warmup_events,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            var_floor=var_floor,
            garch_persistence_max=garch_persistence_max,
            hawkes_intensity_floor=hawkes_intensity_floor,
        )

    events = df_to_events(data)
    builder = FIBBuilder(config)
    bars: list[FIBBar] = []

    for ev in events:
        bar = builder.update(ev)
        if bar is not None:
            bars.append(bar)

    final = builder.flush()
    if final is not None:
        bars.append(final)

    return bars_to_df(bars)


# ── Baseline bar builders ────────────────────────────────────────────────────

def build_time_bars(
    data: pd.DataFrame,
    seconds_per_bar: float = 60.0,
) -> pd.DataFrame:
    """Build fixed-duration time bars."""
    events = df_to_events(data)
    bars = build_time_bars_from_events(events, seconds_per_bar=seconds_per_bar)
    return bars_to_df(bars)


def build_tick_bars(
    data: pd.DataFrame,
    ticks_per_bar: int = 100,
) -> pd.DataFrame:
    """Build fixed-tick-count bars."""
    events = df_to_events(data)
    bars = build_tick_bars_from_events(events, ticks_per_bar=ticks_per_bar)
    return bars_to_df(bars)


def build_volume_bars(
    data: pd.DataFrame,
    volume_per_bar: float = 1000.0,
) -> pd.DataFrame:
    """Build fixed-volume bars."""
    events = df_to_events(data)
    bars = build_volume_bars_from_events(events, volume_per_bar=volume_per_bar)
    return bars_to_df(bars)


def build_dollar_bars(
    data: pd.DataFrame,
    dollar_per_bar: float = 100_000.0,
) -> pd.DataFrame:
    """Build fixed-dollar-value bars."""
    events = df_to_events(data)
    bars = build_dollar_bars_from_events(events, dollar_per_bar=dollar_per_bar)
    return bars_to_df(bars)
