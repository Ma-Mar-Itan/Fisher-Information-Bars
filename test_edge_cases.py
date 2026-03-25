"""Edge-case tests: singular matrices, sparse events, missing fields, short data."""
import numpy as np
import pandas as pd
import pytest
from fibars import build_fib_bars, FIBConfig
from fibars.bars.fib_builder import FIBBuilder
from fibars.events import MarketEvent
from fibars.information.scalarizers import LogdetScalarizer
from fibars.utils.math import ridge_logdet, symmetrize


# ── Singular information matrix ─────────────────────────────────────────────

def test_logdet_zero_matrix_stable():
    J = np.zeros((2, 2))
    val = ridge_logdet(J, eps=1e-6)
    assert np.isfinite(val)


def test_logdet_negative_eigenvalue_stable():
    # Artificially create a non-PSD matrix
    J = np.array([[1.0, 2.0], [2.0, 1.0]])   # eigenvalues -1, 3
    val = ridge_logdet(J, eps=1e-4)
    assert np.isfinite(val)


def test_symmetrize():
    A = np.array([[1.0, 3.0], [0.0, 2.0]])
    S = symmetrize(A)
    assert np.allclose(S, S.T)
    assert np.isclose(S[0, 1], 1.5)


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


# ── Missing bid/ask ──────────────────────────────────────────────────────────

def test_missing_bid_ask_spread_none():
    df = pd.DataFrame({
        "timestamp": np.arange(50, dtype=float),
        "price": 100.0 + np.random.default_rng(0).normal(0, 0.1, 50),
        "size": np.ones(50),
    })
    result = build_fib_bars(df, timeout_seconds=10.0)
    # mean_spread should be None/NaN when no bid/ask provided
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
    # Should produce one bar (from flush) or empty
    assert isinstance(result, pd.DataFrame)


def test_two_events():
    df = pd.DataFrame({"timestamp": [0.0, 1.0], "price": [100.0, 100.1]})
    result = build_fib_bars(df, timeout_seconds=2.0)
    assert isinstance(result, pd.DataFrame)


# ── Sparse events (large gaps) ───────────────────────────────────────────────

def test_sparse_events_timeout_fires():
    df = pd.DataFrame({
        "timestamp": np.arange(0, 2000, 200, dtype=float),  # 10 events, 200s apart
        "price": 100.0 + np.arange(10) * 0.1,
        "size": np.ones(10),
    })
    result = build_fib_bars(df, timeout_seconds=250.0, eta=1e6)
    # With huge eta, only timeouts should close bars
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
    from fibars.data.adapters import df_to_events
    for e in df_to_events(df):
        b = builder.update(e)
        if b is not None:
            bars.append(b)
    b = builder.flush()
    if b:
        bars.append(b)
    assert all(b.n_events <= 51 for b in bars)  # +1 tolerance for open event


# ── Dollar bars missing size ─────────────────────────────────────────────────

def test_dollar_bars_zero_size():
    df = pd.DataFrame({
        "timestamp": np.arange(100, dtype=float),
        "price": np.full(100, 100.0),
        "size": np.zeros(100),   # zero volume
    })
    from fibars import build_dollar_bars
    result = build_dollar_bars(df, dollar_per_bar=1000.0)
    # Should produce one leftover bar (never reaches threshold)
    assert isinstance(result, pd.DataFrame)
