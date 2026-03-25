"""Tests for baseline bar builders."""
import numpy as np
import pandas as pd
import pytest
from fibars import build_time_bars, build_tick_bars, build_volume_bars, build_dollar_bars


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.1, n)),
        "size": rng.integers(1, 10, n).astype(float),
    })


class TestTimeBars:

    def test_produces_bars(self):
        df = make_df(200)
        result = build_time_bars(df, seconds_per_bar=50.0)
        assert len(result) >= 1

    def test_bar_duration_close_to_target(self):
        df = make_df(500)
        result = build_time_bars(df, seconds_per_bar=50.0)
        # Most bars (except last) should span roughly 50 seconds
        non_last = result.iloc[:-1]
        if len(non_last) > 0:
            assert (non_last["duration_seconds"] <= 50.0 + 1.0).all()

    def test_ohlc_valid(self):
        df = make_df(200)
        result = build_time_bars(df, seconds_per_bar=30.0)
        assert (result["high"] >= result["low"]).all()
        assert (result["n_events"] >= 1).all()

    def test_scalarizer_name_tag(self):
        df = make_df()
        result = build_time_bars(df, seconds_per_bar=50.0)
        assert (result["scalarizer_name"] == "time").all()


class TestTickBars:

    def test_exact_tick_count(self):
        df = make_df(200)
        ticks = 25
        result = build_tick_bars(df, ticks_per_bar=ticks)
        # All complete bars (not last) should have exactly `ticks` events
        non_last = result.iloc[:-1]
        if len(non_last) > 0:
            assert (non_last["n_events"] == ticks).all()

    def test_last_bar_is_remainder(self):
        n = 210
        df = make_df(n)
        result = build_tick_bars(df, ticks_per_bar=50)
        total_events = result["n_events"].sum()
        assert total_events == n

    def test_scalarizer_name_tag(self):
        df = make_df()
        result = build_tick_bars(df, ticks_per_bar=50)
        assert (result["scalarizer_name"] == "tick").all()


class TestVolumeBars:

    def test_cumulative_volume_correct(self):
        df = make_df(300)
        result = build_volume_bars(df, volume_per_bar=100.0)
        assert len(result) >= 1
        assert result["sum_volume"].notna().all()

    def test_zero_volume_produces_one_bar(self):
        df = pd.DataFrame({
            "timestamp": np.arange(50, dtype=float),
            "price": np.full(50, 100.0),
            "size": np.zeros(50),
        })
        result = build_volume_bars(df, volume_per_bar=100.0)
        assert len(result) == 1  # never reaches threshold → flush at end

    def test_scalarizer_name_tag(self):
        df = make_df()
        result = build_volume_bars(df, volume_per_bar=100.0)
        assert (result["scalarizer_name"] == "volume").all()


class TestDollarBars:

    def test_produces_bars(self):
        df = make_df(300)
        result = build_dollar_bars(df, dollar_per_bar=500.0)
        assert len(result) >= 1

    def test_dollar_value_column_populated(self):
        df = make_df(200)
        result = build_dollar_bars(df, dollar_per_bar=200.0)
        assert result["dollar_value"].notna().all()
        assert (result["dollar_value"] >= 0).all()

    def test_all_events_covered(self):
        n = 300
        df = make_df(n)
        result = build_dollar_bars(df, dollar_per_bar=200.0)
        total = result["n_events"].sum()
        assert total == n

    def test_scalarizer_name_tag(self):
        df = make_df()
        result = build_dollar_bars(df, dollar_per_bar=500.0)
        assert (result["scalarizer_name"] == "dollar").all()


class TestCommonSchema:
    """All bar types share the same FIBBar output schema."""

    @pytest.mark.parametrize("builder,kwargs", [
        (build_time_bars,   {"seconds_per_bar": 50.0}),
        (build_tick_bars,   {"ticks_per_bar": 50}),
        (build_volume_bars, {"volume_per_bar": 200.0}),
        (build_dollar_bars, {"dollar_per_bar": 5000.0}),
    ])
    def test_required_columns_present(self, builder, kwargs):
        df = make_df(200)
        result = builder(df, **kwargs)
        required = [
            "open_time", "close_time", "duration_seconds", "n_events",
            "open", "high", "low", "close",
            "sum_volume", "dollar_value",
            "information_scalar", "threshold_at_close", "timeout_flag",
            "model_name", "info_mode", "scalarizer_name",
            "start_event_index", "end_event_index",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"
