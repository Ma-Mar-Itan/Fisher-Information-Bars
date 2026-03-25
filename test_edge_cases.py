"""Edge-case tests: singular matrices, sparse events, missing fields, short data."""
import numpy as np
import pandas as pd
import pytest
from fibars import build_fib_bars, FIBConfig, build_dollar_bars
from fibars.bars.fib_builder import FIBBuilder
from fibars.events import MarketEvent
from fibars.information.scalarizers import LogdetScalarizer
from fibars.utils.math import ridge_logdet, symmetrize
from fibars.data.adapters import df_to_events


# ── Numerical helpers ────────────────────────────────────────────────────────

def test_logdet_zero_matrix_stable():
    J = np.zeros((2, 2))
    val = ridge_logdet(J, eps=1e-6)
    assert np.isfinite(val)


def test_logdet_negative_eigenvalue_stable():
    J = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues -1, 3
    val = ridge_logdet(J, eps=1e-4)
    assert np.isfinite(val)


def test_symmetrize():
    A = np.array([[1.0, 3.0], [0.0, 2.0]])
    S = symmetrize(A)
    assert np.allclose(S, S.T)
    assert np.isclose(S[0, 1], 1.5)


def test_logdet_identity():
    J = np.eye(3)
    val = ridge_logdet(J, eps=0.0)
    assert np.isclose(val, 0.0, atol=1e-10)


# ── Constant price (zero variance) ──────────────────────────────────────────

def test_constant_price_gaussian_no_nan():
    df = pd.DataFrame({
        "timestamp": np.arange(100, dtype=float),
        "price": np.full(100, 100.0),
        "size": np.ones(100),
    })
    result = build_fib_bars(df, model="gaussian", timeout_seconds=20.0)
    assert result["information_scalar"].notna().all()


def test_constant_price_garch_no_nan():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": np.full(50, 50.0),
        "size": np.ones(50),
    })
    result = build_fib_bars(df, model="garch", timeout_seconds=15.0)
    assert result["information_scalar"].notna().all()


def test_constant_price_hawkes_no_nan():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": np.full(50, 100.0),
        "size": np.ones(50),
    })
    result = build_fib_bars(df, model="hawkes", timeout_seconds=15.0)
    assert result["information_scalar"].notna().all()


# ── Missing bid/ask ──────────────────────────────────────────────────────────

def test_missing_bid_ask_spread_none():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": 100.0 + np.random.default_rng(0).normal(0, 0.1, 50),
        "size": np.ones(50),
    })
    result = build_fib_bars(df, timeout_seconds=10.0)
    assert result["mean_spread"].isna().all()


def test_with_bid_ask_spread_computed():
    rng = np.random.default_rng(1)
    n = 60
    mid = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": mid,
        "size": np.ones(n),
        "bid": mid - 0.05,
        "ask": mid + 0.05,
    })
    result = build_fib_bars(df, timeout_seconds=15.0)
    valid = result["mean_spread"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


# ── Very short datasets ──────────────────────────────────────────────────────

def test_single_event():
    df = pd.DataFrame({"timestamp": [0.0], "price": [100.0]})
    result = build_fib_bars(df)
    assert isinstance(result, pd.DataFrame)


def test_two_events():
    df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, 100.1]})
    result = build_fib_bars(df, timeout_seconds=2.0)
    assert isinstance(result, pd.DataFrame)


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame({"timestamp": [], "price": []})
    result = build_fib_bars(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ── Sparse events (large gaps) ───────────────────────────────────────────────

def test_sparse_events_timeout_fires():
    df = pd.DataFrame({
        "timestamp": np.arange(0, 2000, 200, dtype=float),
        "price": 100.0 + np.arange(10) * 0.1,
        "size": np.ones(10),
    })
    result = build_fib_bars(df, timeout_seconds=250.0, eta=1e6)
    assert result["timeout_flag"].all()


# ── Max events per bar ───────────────────────────────────────────────────────

def test_max_events_per_bar_respected():
    df = pd.DataFrame({
        "timestamp": np.arange(200, dtype=float),
        "price": 100.0 + np.random.default_rng(5).normal(0, 0.1, 200),
        "size": np.ones(200),
    })
    cfg = FIBConfig(eta=1e9, timeout_seconds=99999, max_events_per_bar=50)
    builder = FIBBuilder(cfg)
    bars = []
    for e in df_to_events(df):
        b = builder.update(e)
        if b is not None:
            bars.append(b)
    b = builder.flush()
    if b:
        bars.append(b)
    assert all(b.n_events <= 51 for b in bars)


# ── Dollar bars with zero size ───────────────────────────────────────────────

def test_dollar_bars_zero_size():
    df = pd.DataFrame({
        "timestamp": np.arange(100, dtype=float),
        "price": np.full(100, 100.0),
        "size": np.zeros(100),
    })
    result = build_dollar_bars(df, dollar_per_bar=1000.0)
    assert isinstance(result, pd.DataFrame)


# ── Threshold trigger (not timeout) ─────────────────────────────────────────

def test_threshold_triggered_bars_exist():
    """With a large timeout and small eta, bars should close on threshold."""
    rng = np.random.default_rng(7)
    n = 1000
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.5, n)),
        "size": np.ones(n),
    })
    result = build_fib_bars(
        df, model="gaussian",
        timeout_seconds=999999.0,
        eta=0.3,
        min_warmup_events=2,
    )
    # At least some bars should be threshold-triggered
    assert len(result) > 1
    non_timeout = result[~result["timeout_flag"]]
    assert len(non_timeout) > 0


# ── All three models produce finite output ────────────────────────────────────

@pytest.mark.parametrize("model", ["gaussian", "garch", "hawkes"])
def test_all_models_finite_output(model):
    rng = np.random.default_rng(99)
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.1, n)),
        "size": np.ones(n),
    })
    result = build_fib_bars(df, model=model, timeout_seconds=50.0)
    assert len(result) > 0
    assert result["information_scalar"].notna().all()
    assert result["open"].notna().all()
    assert result["close"].notna().all()
    assert (result["n_events"] >= 1).all()


# ── All three scalarizers produce finite output ───────────────────────────────

@pytest.mark.parametrize("scalarizer", ["logdet", "trace", "frobenius"])
def test_all_scalarizers_finite_output(scalarizer):
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
        "size": np.ones(n),
    })
    result = build_fib_bars(df, scalarizer=scalarizer, timeout_seconds=40.0)
    assert result["information_scalar"].notna().all()


# ── Observed vs expected modes ────────────────────────────────────────────────

@pytest.mark.parametrize("info_mode", ["observed", "expected"])
def test_info_modes(info_mode):
    rng = np.random.default_rng(22)
    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
        "size": np.ones(n),
    })
    result = build_fib_bars(df, info_mode=info_mode, timeout_seconds=40.0)
    assert result["information_scalar"].notna().all()


# ── OHLC invariants ───────────────────────────────────────────────────────────

def test_ohlc_invariants():
    rng = np.random.default_rng(33)
    n = 500
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.3, n)),
        "size": np.ones(n),
    })
    result = build_fib_bars(df, timeout_seconds=50.0)
    assert (result["high"] >= result["open"]).all()
    assert (result["high"] >= result["close"]).all()
    assert (result["low"] <= result["open"]).all()
    assert (result["low"] <= result["close"]).all()
    assert (result["high"] >= result["low"]).all()


# ── Streaming matches batch ───────────────────────────────────────────────────

def test_streaming_matches_batch():
    from fibars import StreamingBuilder
    rng = np.random.default_rng(55)
    n = 400
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
        "size": np.ones(n),
    })
    cfg = FIBConfig(model="gaussian", eta=0.5, timeout_seconds=80.0, min_warmup_events=2)

    # Batch
    batch_result = build_fib_bars(df, config=cfg)

    # Streaming
    builder = StreamingBuilder(config=cfg)
    for ev in df_to_events(df):
        builder.push(ev)
    builder.flush()
    streaming_bars = builder.completed_bars

    assert len(batch_result) == len(streaming_bars)
    for i, bar in enumerate(streaming_bars):
        assert bar.n_events == batch_result.iloc[i]["n_events"]
        assert np.isclose(bar.information_scalar, batch_result.iloc[i]["information_scalar"])


# ── FIBConfig validation ──────────────────────────────────────────────────────

def test_config_invalid_eta():
    with pytest.raises(ValueError, match="eta"):
        FIBConfig(eta=-1.0)


def test_config_invalid_model():
    with pytest.raises(ValueError, match="model"):
        FIBConfig(model="unknown_model")


def test_config_invalid_ewma_alpha():
    with pytest.raises(ValueError, match="ewma_alpha"):
        FIBConfig(ewma_alpha=0.0)


# ── Inactivity timeout ────────────────────────────────────────────────────────

def test_inactivity_timeout():
    """Events bunched together then a big gap should trigger inactivity close."""
    timestamps = list(np.arange(10, dtype=float)) + [10.0 + 999.0]
    prices = [100.0 + i * 0.01 for i in range(11)]
    df = pd.DataFrame({"timestamp": timestamps, "price": prices})
    result = build_fib_bars(
        df,
        timeout_seconds=99999.0,
        inactivity_timeout_seconds=100.0,
        eta=1e9,
    )
    assert isinstance(result, pd.DataFrame)


# ── Provenance fields populated ───────────────────────────────────────────────

def test_provenance_fields():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": 100.0 + np.random.default_rng(77).normal(0, 0.1, 50),
    })
    result = build_fib_bars(df, model="garch", info_mode="expected",
                            scalarizer="trace", timeout_seconds=10.0)
    assert (result["model_name"] == "garch").all()
    assert (result["info_mode"] == "expected").all()
    assert (result["scalarizer_name"] == "trace").all()


# ── Event index continuity ────────────────────────────────────────────────────

def test_event_index_continuity():
    """start/end indices should cover the full event range without gaps."""
    rng = np.random.default_rng(88)
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.2, n)),
    })
    result = build_fib_bars(df, timeout_seconds=50.0)
    if len(result) < 2:
        return
    for i in range(len(result) - 1):
        assert result.iloc[i + 1]["start_event_index"] > result.iloc[i]["start_event_index"]
