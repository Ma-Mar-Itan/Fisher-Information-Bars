"""Tests for DataFrame ↔ MarketEvent adapters."""
import numpy as np
import pandas as pd
import pytest
from fibars.data.adapters import df_to_events, df_to_event_stream, bars_to_df
from fibars.events import MarketEvent


def make_full_df(n=50):
    rng = np.random.default_rng(0)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": mid,
        "size": rng.integers(1, 5, n).astype(float),
        "bid": mid - 0.02,
        "ask": mid + 0.02,
        "side": np.where(rng.random(n) > 0.5, "B", "S"),
        "event_type": ["trade"] * n,
    })


class TestDfToEvents:

    def test_basic_conversion(self):
        df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, 100.1]})
        events = df_to_events(df)
        assert len(events) == 2
        assert all(isinstance(e, MarketEvent) for e in events)

    def test_all_optional_fields_mapped(self):
        df = make_full_df(10)
        events = df_to_events(df)
        assert events[0].bid is not None
        assert events[0].ask is not None
        assert events[0].side in ("B", "S")
        assert events[0].event_type == "trade"

    def test_missing_size_defaults_to_zero(self):
        df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, 100.1]})
        events = df_to_events(df)
        assert all(e.size == 0.0 for e in events)

    def test_nan_bid_becomes_none(self):
        df = pd.DataFrame({
            "timestamp": [0.0, 1.0],
            "price": [100.0, 100.1],
            "bid": [float("nan"), 99.9],
            "ask": [float("nan"), 100.1],
        })
        events = df_to_events(df)
        assert events[0].bid is None
        assert events[1].bid == pytest.approx(99.9)

    def test_unsorted_timestamps_sorted(self):
        df = pd.DataFrame({
            "timestamp": [2.0, 0.0, 1.0],
            "price": [102.0, 100.0, 101.0],
        })
        events = df_to_events(df)
        ts = [e.timestamp for e in events]
        assert ts == sorted(ts)

    def test_missing_price_raises(self):
        df = pd.DataFrame({"timestamp": [0.0], "foo": [1.0]})
        with pytest.raises(ValueError, match="price"):
            df_to_events(df)

    def test_missing_timestamp_raises(self):
        df = pd.DataFrame({"price": [100.0]})
        with pytest.raises(ValueError, match="timestamp"):
            df_to_events(df)

    def test_nan_price_raises(self):
        df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            df_to_events(df)


class TestDfToEventStream:

    def test_generator_yields_correct_count(self):
        df = make_full_df(20)
        events = list(df_to_event_stream(df))
        assert len(events) == 20

    def test_generator_same_as_list(self):
        df = make_full_df(20)
        list_events = df_to_events(df)
        stream_events = list(df_to_event_stream(df))
        for a, b in zip(list_events, stream_events):
            assert a.timestamp == b.timestamp
            assert a.price == b.price


class TestBarsToDF:

    def test_empty_list_returns_empty_df(self):
        result = bars_to_df([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_columns_present(self):
        from fibars import build_fib_bars
        df = make_full_df(100)
        bars_df = build_fib_bars(df, timeout_seconds=20.0)
        assert "open_time" in bars_df.columns
        assert "close_time" in bars_df.columns
        assert "information_scalar" in bars_df.columns


class TestMarketEventProperties:

    def test_spread_computed(self):
        ev = MarketEvent(timestamp=0.0, price=100.0, bid=99.9, ask=100.1)
        assert pytest.approx(ev.spread) == 0.2

    def test_spread_none_without_quotes(self):
        ev = MarketEvent(timestamp=0.0, price=100.0)
        assert ev.spread is None

    def test_mid_uses_bid_ask(self):
        ev = MarketEvent(timestamp=0.0, price=100.0, bid=99.0, ask=101.0)
        assert ev.mid == 100.0

    def test_mid_fallback_to_price(self):
        ev = MarketEvent(timestamp=0.0, price=100.5)
        assert ev.mid == 100.5
