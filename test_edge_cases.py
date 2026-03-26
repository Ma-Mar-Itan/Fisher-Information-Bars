"""Edge-case tests: numerical stability, close reasons, inactivity, augmentation."""
import numpy as np
import pandas as pd
import pytest
from fibars import (
    build_fib_bars, build_fib_bars_from_events,
    augment_with_fib_features, FIBConfig, StreamingBuilder,
    build_dollar_bars,
)
from fibars.bars.fib_builder import FIBBuilder
from fibars.events import MarketEvent
from fibars.information.scalarizers import LogdetScalarizer
from fibars.utils.math import ridge_logdet, symmetrize, safe_outer
from fibars.data.adapters import df_to_events


# ── Numerical helpers ────────────────────────────────────────────────────────

def test_logdet_zero_matrix_stable():
    val = ridge_logdet(np.zeros((2, 2)), eps=1e-6)
    assert np.isfinite(val)


def test_logdet_negative_eigenvalue_stable():
    J = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues -1, 3
    val = ridge_logdet(J, eps=1e-4)
    assert np.isfinite(val)


def test_logdet_identity():
    val = ridge_logdet(np.eye(3), eps=0.0)
    assert np.isclose(val, 0.0, atol=1e-8)


def test_symmetrize():
    A = np.array([[1.0, 3.0], [0.0, 2.0]])
    S = symmetrize(A)
    assert np.allclose(S, S.T)
    assert np.isclose(S[0, 1], 1.5)


def test_safe_outer_nan_returns_zeros():
    s = np.array([float("nan"), 1.0])
    J = safe_outer(s)
    assert np.all(J == 0.0)


def test_safe_outer_inf_returns_zeros():
    s = np.array([float("inf"), 1.0])
    J = safe_outer(s)
    assert np.all(J == 0.0)


# ── Constant price ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["gaussian", "garch", "hawkes"])
def test_constant_price_no_nan(model):
    df = pd.DataFrame({
        "timestamp": np.arange(80, dtype=float),
        "price": np.full(80, 100.0),
        "size": np.ones(80),
    })
    result = build_fib_bars(df, model=model, timeout_seconds=15.0)
    assert result["information_scalar"].notna().all()
    assert len(result) > 0


# ── Missing bid/ask ───────────────────────────────────────────────────────────

def test_missing_bid_ask_spread_none():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": 100.0 + np.random.default_rng(0).normal(0, 0.1, 50),
    })
    result = build_fib_bars(df, timeout_seconds=10.0)
    assert result["mean_spread"].isna().all()


def test_with_bid_ask_spread_positive():
    rng = np.random.default_rng(1)
    n = 60
    mid = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": mid, "size": np.ones(n),
        "bid": mid - 0.05, "ask": mid + 0.05,
    })
    result = build_fib_bars(df, timeout_seconds=15.0)
    valid = result["mean_spread"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


# ── Short datasets ────────────────────────────────────────────────────────────

def test_single_event_returns_df():
    df = pd.DataFrame({"timestamp": [0.0], "price": [100.0]})
    result = build_fib_bars(df)
    assert isinstance(result, pd.DataFrame)


def test_two_events_no_crash():
    df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, 100.1]})
    result = build_fib_bars(df, timeout_seconds=2.0)
    assert isinstance(result, pd.DataFrame)


def test_empty_dataframe():
    df = pd.DataFrame({"timestamp": [], "price": []})
    result = build_fib_bars(df)
    assert len(result) == 0


# ── Close reason labeling ────────────────────────────────────────────────────

def test_close_reason_threshold():
    """With large timeout and volatile prices, some bars close on threshold."""
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.5, n)),
    })
    result = build_fib_bars(df, timeout_seconds=999999.0, eta=0.2, min_warmup_events=2)
    assert "threshold" in result["close_reason"].values


def test_close_reason_timeout():
    rng = np.random.default_rng(8)
    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.01, n)),
    })
    result = build_fib_bars(df, timeout_seconds=30.0, eta=1e8)
    assert "timeout" in result["close_reason"].values


def test_close_reason_flush():
    rng = np.random.default_rng(9)
    n = 50
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.01, n)),
    })
    result = build_fib_bars(df, timeout_seconds=9999.0, eta=1e8)
    assert result.iloc[-1]["close_reason"] == "flush"


def test_close_reason_inactivity():
    """Gap of 1000s between events with inactivity_timeout=100 should fire."""
    timestamps = list(np.arange(10, dtype=float)) + [10.0 + 1000.0]
    prices = [100.0 + i * 0.01 for i in range(11)]
    df = pd.DataFrame({"timestamp": timestamps, "price": prices})
    result = build_fib_bars(
        df,
        timeout_seconds=99999.0,
        inactivity_timeout_seconds=100.0,
        eta=1e9,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # At least one bar should be inactivity-triggered
    assert "inactivity" in result["close_reason"].values


def test_close_reason_max_events():
    # Constant price → zero variance → zero scalar → never threshold-triggers
    # So max_events cap must fire
    df = pd.DataFrame({
        "timestamp": np.arange(200, dtype=float),
        "price": np.full(200, 100.0),  # constant — no information accumulates
        "size": np.ones(200),
    })
    cfg = FIBConfig(eta=1e9, timeout_seconds=99999, max_events_per_bar=50)
    result = build_fib_bars(df, config=cfg)
    assert "max_events" in result["close_reason"].values


def test_timeout_flag_consistent_with_close_reason():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.3, n)),
    })
    result = build_fib_bars(df, timeout_seconds=40.0, eta=0.3, min_warmup_events=2)
    # timeout_flag should be False iff close_reason == "threshold"
    for _, row in result.iterrows():
        if row["close_reason"] == "threshold":
            assert not row["timeout_flag"]
        else:
            assert row["timeout_flag"]


# ── Inactivity detection in batch mode ───────────────────────────────────────

def test_inactivity_actually_triggers():
    """Batch mode: gap between consecutive events triggers inactivity close."""
    # First 5 events are 1s apart, then a 500s gap
    ts = list(np.arange(5, dtype=float)) + [5.0 + 500.0, 5.0 + 501.0]
    prices = [100.0 + i * 0.01 for i in range(7)]
    df = pd.DataFrame({"timestamp": ts, "price": prices})
    result = build_fib_bars(
        df,
        inactivity_timeout_seconds=60.0,
        timeout_seconds=99999.0,
        eta=1e9,
        min_warmup_events=1,
    )
    # The event at t=505 arrives after a 500s gap → inactivity bar
    reasons = result["close_reason"].values
    assert "inactivity" in reasons


# ── Max events cap ────────────────────────────────────────────────────────────

def test_max_events_per_bar_hard_cap():
    df = pd.DataFrame({
        "timestamp": np.arange(300, dtype=float),
        "price": 100.0 + np.random.default_rng(5).normal(0, 0.1, 300),
    })
    cfg = FIBConfig(eta=1e9, timeout_seconds=99999, max_events_per_bar=40)
    result = build_fib_bars(df, config=cfg)
    # All complete bars must respect the cap
    assert (result.iloc[:-1]["n_events"] <= 40).all()


# ── Parameter dimension from model ───────────────────────────────────────────

def test_n_params_from_model_not_hardcoded():
    """FIBBuilder must get n_params from model, not a lookup dict."""
    for model_name, expected in [("gaussian", 2), ("garch", 3), ("hawkes", 3)]:
        cfg = FIBConfig(model=model_name)
        builder = FIBBuilder(cfg)
        assert builder._n_params == expected
        assert builder._J.shape == (expected, expected)


# ── Augmented features ────────────────────────────────────────────────────────

def test_augmented_features_present():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": np.arange(300, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, 300)),
    })
    bars = build_fib_bars(df, timeout_seconds=40.0)
    aug = augment_with_fib_features(bars)
    for col in [
        "threshold_utilization", "information_rate", "log_duration",
        "log_n_events", "log_information_scalar", "price_range",
        "price_range_pct", "vwap", "is_threshold_close", "bar_index",
    ]:
        assert col in aug.columns, f"Missing: {col}"


def test_augmented_no_inf():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "timestamp": np.arange(300, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, 300)),
        "size": rng.integers(1, 10, 300).astype(float),
    })
    bars = build_fib_bars(df, timeout_seconds=40.0)
    aug = augment_with_fib_features(bars)
    numeric_cols = aug.select_dtypes(include=[np.number]).columns
    assert not aug[numeric_cols].isin([np.inf, -np.inf]).any().any()


def test_augmented_empty_df():
    aug = augment_with_fib_features(pd.DataFrame())
    assert isinstance(aug, pd.DataFrame)


# ── OHLC invariants ───────────────────────────────────────────────────────────

def test_ohlc_invariants():
    rng = np.random.default_rng(33)
    n = 500
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.3, n)),
    })
    result = build_fib_bars(df, timeout_seconds=50.0)
    assert (result["high"] >= result["open"]).all()
    assert (result["high"] >= result["close"]).all()
    assert (result["low"] <= result["open"]).all()
    assert (result["low"] <= result["close"]).all()
    assert (result["high"] >= result["low"]).all()


# ── Streaming matches batch ───────────────────────────────────────────────────

def test_streaming_matches_batch():
    rng = np.random.default_rng(55)
    n = 400
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
    })
    cfg = FIBConfig(model="gaussian", eta=0.5, timeout_seconds=80.0, min_warmup_events=2)

    batch_result = build_fib_bars(df, config=cfg)

    builder = StreamingBuilder(config=cfg)
    for ev in df_to_events(df):
        builder.push(ev)
    builder.flush()
    streaming_bars = builder.completed_bars

    assert len(batch_result) == len(streaming_bars)
    for i, bar in enumerate(streaming_bars):
        assert bar.n_events == batch_result.iloc[i]["n_events"]
        assert np.isclose(bar.information_scalar, batch_result.iloc[i]["information_scalar"])


# ── All models × all scalarizers × both modes ─────────────────────────────────

@pytest.mark.parametrize("model", ["gaussian", "garch", "hawkes"])
@pytest.mark.parametrize("scalarizer", ["logdet", "trace", "frobenius"])
@pytest.mark.parametrize("info_mode", ["observed", "expected"])
def test_all_combinations_finite(model, scalarizer, info_mode):
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.1, n)),
    })
    result = build_fib_bars(
        df, model=model, scalarizer=scalarizer,
        info_mode=info_mode, timeout_seconds=30.0, min_warmup_events=2,
    )
    assert len(result) > 0
    assert result["information_scalar"].notna().all()
    assert np.isfinite(result["information_scalar"]).all()


# ── FIBConfig validation ──────────────────────────────────────────────────────

class TestFIBConfigValidation:
    def test_negative_eta(self):
        with pytest.raises(ValueError, match="eta"):
            FIBConfig(eta=-1.0)

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="model"):
            FIBConfig(model="xgboost")

    def test_zero_ewma_alpha(self):
        with pytest.raises(ValueError, match="ewma_alpha"):
            FIBConfig(ewma_alpha=0.0)

    def test_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            FIBConfig(timeout_seconds=-1.0)

    def test_zero_max_events(self):
        with pytest.raises(ValueError, match="max_events_per_bar"):
            FIBConfig(max_events_per_bar=0)

    def test_negative_inactivity(self):
        with pytest.raises(ValueError, match="inactivity"):
            FIBConfig(inactivity_timeout_seconds=-5.0)

    def test_min_above_max_threshold(self):
        with pytest.raises(ValueError, match="min_threshold"):
            FIBConfig(min_threshold=10.0, max_threshold=5.0)

    def test_invalid_var_floor(self):
        with pytest.raises(ValueError, match="var_floor"):
            FIBConfig(var_floor=0.0)

    def test_garch_persistence_out_of_range(self):
        with pytest.raises(ValueError, match="garch_persistence"):
            FIBConfig(garch_persistence_max=1.5)

    def test_valid_config_no_raise(self):
        cfg = FIBConfig(model="garch", eta=2.0, timeout_seconds=120.0)
        assert cfg.model == "garch"


# ── Event index continuity ────────────────────────────────────────────────────

def test_event_index_monotone():
    rng = np.random.default_rng(88)
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
    })
    result = build_fib_bars(df, timeout_seconds=50.0)
    if len(result) < 2:
        return
    assert (result["start_event_index"].diff().dropna() > 0).all()


def test_total_events_equals_input():
    rng = np.random.default_rng(77)
    n = 500
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
    })
    result = build_fib_bars(df, timeout_seconds=50.0, eta=0.5, min_warmup_events=2)
    # Total events across all bars should equal input (minus first event of each bar
    # which is counted when bar opens — n_events includes the opening event)
    # Actually: every event belongs to exactly one bar
    assert result["n_events"].sum() == n
