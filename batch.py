"""Batch API: build bars from a complete DataFrame in one call."""
from __future__ import annotations
import numpy as np
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
    config : FIBConfig | None
        If provided, all other keyword args are ignored.

    Returns
    -------
    pd.DataFrame  — one row per completed bar.
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


def build_fib_bars_from_events(
    events: list,
    config: FIBConfig | None = None,
) -> pd.DataFrame:
    """Build FIB bars directly from a list of MarketEvent objects."""
    cfg = config or FIBConfig()
    builder = FIBBuilder(cfg)
    bars: list[FIBBar] = []
    for ev in events:
        bar = builder.update(ev)
        if bar is not None:
            bars.append(bar)
    final = builder.flush()
    if final is not None:
        bars.append(final)
    return bars_to_df(bars)


def augment_with_fib_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a FIB bars DataFrame with derived features for ML/research.

    Added columns
    -------------
    threshold_utilization  : information_scalar / threshold_at_close
    information_rate       : information_scalar / max(duration_seconds, 1e-3)
    log_duration           : log(1 + duration_seconds)
    log_n_events           : log(1 + n_events)
    log_information_scalar : log(1 + information_scalar)
    price_range            : high - low
    price_range_pct        : (high - low) / open * 100  (if open != 0)
    vwap                   : dollar_value / max(sum_volume, 1e-12)
    is_threshold_close     : bool — closed on threshold (not timeout/flush)
    bar_index              : 0-based sequential bar number
    """
    if bars_df.empty:
        return bars_df.copy()

    df = bars_df.copy()

    # Threshold utilization — ratio of accumulated info to quantum
    thr = df["threshold_at_close"].clip(lower=1e-12)
    df["threshold_utilization"] = df["information_scalar"] / thr

    # Information rate per second
    dur = df["duration_seconds"].clip(lower=1e-3)
    df["information_rate"] = df["information_scalar"] / dur

    # Log-transformed features
    df["log_duration"] = np.log1p(df["duration_seconds"])
    df["log_n_events"] = np.log1p(df["n_events"])
    df["log_information_scalar"] = np.log1p(df["information_scalar"].clip(lower=0))

    # Price-based
    df["price_range"] = df["high"] - df["low"]
    open_safe = df["open"].replace(0, np.nan)
    df["price_range_pct"] = (df["price_range"] / open_safe * 100).fillna(0.0)

    # VWAP
    vol_safe = df["sum_volume"].clip(lower=1e-12)
    df["vwap"] = df["dollar_value"] / vol_safe

    # Close reason boolean
    df["is_threshold_close"] = df["close_reason"] == "threshold"

    # Bar index
    df["bar_index"] = np.arange(len(df))

    return df


def build_baseline_bars(
    data: pd.DataFrame,
    bar_type: str = "time",
    seconds_per_bar: float = 60.0,
    ticks_per_bar: int = 100,
    volume_per_bar: float = 1000.0,
    dollar_per_bar: float = 100_000.0,
) -> pd.DataFrame:
    """
    Build a single baseline bar type. bar_type: 'time'|'tick'|'volume'|'dollar'.
    """
    events = df_to_events(data)
    if bar_type == "time":
        bars = build_time_bars_from_events(events, seconds_per_bar=seconds_per_bar)
    elif bar_type == "tick":
        bars = build_tick_bars_from_events(events, ticks_per_bar=ticks_per_bar)
    elif bar_type == "volume":
        bars = build_volume_bars_from_events(events, volume_per_bar=volume_per_bar)
    elif bar_type == "dollar":
        bars = build_dollar_bars_from_events(events, dollar_per_bar=dollar_per_bar)
    else:
        raise ValueError(f"Unknown bar_type '{bar_type}'. Choose: time, tick, volume, dollar")
    return bars_to_df(bars)


# ── Individual baseline builders (kept for API compatibility) ────────────────

def build_time_bars(data: pd.DataFrame, seconds_per_bar: float = 60.0) -> pd.DataFrame:
    return build_baseline_bars(data, "time", seconds_per_bar=seconds_per_bar)


def build_tick_bars(data: pd.DataFrame, ticks_per_bar: int = 100) -> pd.DataFrame:
    return build_baseline_bars(data, "tick", ticks_per_bar=ticks_per_bar)


def build_volume_bars(data: pd.DataFrame, volume_per_bar: float = 1000.0) -> pd.DataFrame:
    return build_baseline_bars(data, "volume", volume_per_bar=volume_per_bar)


def build_dollar_bars(data: pd.DataFrame, dollar_per_bar: float = 100_000.0) -> pd.DataFrame:
    return build_baseline_bars(data, "dollar", dollar_per_bar=dollar_per_bar)
