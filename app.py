"""
Fisher Information Bars — Streamlit Demo
=========================================
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fisher Information Bars",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #0e1117;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #e8e8e8; margin-top: 0.2rem; }
    .metric-sub   { font-size: 0.75rem; color: #aaa; margin-top: 0.1rem; }
    .section-header {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #aaa;
        border-bottom: 1px solid #2a2d3e; padding-bottom: 0.3rem; margin-bottom: 0.8rem;
    }
    div[data-testid="stExpander"] { border: 1px solid #2a2d3e !important; border-radius: 6px; }
    .stDownloadButton > button {
        width: 100%; border-radius: 6px;
        background: #1a1d2e; border: 1px solid #3a3d5e; color: #ccc;
    }
    .stDownloadButton > button:hover { background: #2a2d4e; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ── Import library ────────────────────────────────────────────────────────────
try:
    from fibars import (
        build_fib_bars, augment_with_fib_features,
        build_baseline_bars, FIBConfig,
    )
    LIB_OK = True
except ImportError as e:
    LIB_OK = False
    LIB_ERROR = str(e)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#4e8ef7"
C_ORANGE = "#f79e4e"
C_GREEN  = "#4ef7a0"
C_RED    = "#f74e4e"
C_PURPLE = "#b44ef7"
C_GREY   = "#666"
PLOTLY_TEMPLATE = "plotly_dark"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _metric(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _coerce_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Try to convert a timestamp column to float Unix seconds."""
    df = df.copy()
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        df[col] = s.astype(float)
        return df
    try:
        parsed = pd.to_datetime(s, utc=True)
        df[col] = parsed.astype("int64") / 1e9
        return df
    except Exception:
        pass
    try:
        df[col] = s.astype(float)
    except Exception:
        pass
    return df


def _compute_summary(bars_df: pd.DataFrame) -> dict:
    if bars_df.empty:
        return {}
    n = len(bars_df)
    ne = bars_df["n_events"]
    dur = bars_df["duration_seconds"]
    info = bars_df["information_scalar"]
    thr_count = (bars_df["close_reason"] == "threshold").sum()
    reasons = bars_df["close_reason"].value_counts().to_dict()
    cv_events = (ne.std() / ne.mean()) if ne.mean() > 0 else float("nan")
    return {
        "n_bars": n,
        "avg_events_per_bar": float(ne.mean()),
        "cv_events": float(cv_events),
        "avg_duration_s": float(dur.mean()),
        "median_duration_s": float(dur.median()),
        "pct_threshold": float(thr_count / n * 100) if n > 0 else 0.0,
        "avg_info_scalar": float(info.mean()),
        "median_info_scalar": float(info.median()),
        "close_reasons": reasons,
    }


def _generate_synthetic(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Regime-switching random walk with bursts
    price = 100.0
    prices, ts, sizes = [], [], []
    t = 0.0
    for _ in range(n):
        vol = rng.choice([0.05, 0.3, 0.8], p=[0.7, 0.2, 0.1])
        price += rng.normal(0, vol)
        price = max(price, 1.0)
        dt = rng.exponential(0.5)
        t += dt
        prices.append(round(price, 4))
        ts.append(round(t, 4))
        sizes.append(int(rng.integers(1, 20)))
    df = pd.DataFrame({"timestamp": ts, "price": prices, "size": sizes})
    half = 0.02 + 0.01 * rng.random(n)
    df["bid"] = df["price"] - half
    df["ask"] = df["price"] + half
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Fisher Information Bars")
    st.markdown("*Information-geometric financial sampling*")
    st.divider()

    # ── Data source ───────────────────────────────────────────────────────────
    _section("DATA SOURCE")
    data_source = st.radio(
        "Source", ["Upload CSV", "Generate synthetic data"],
        label_visibility="collapsed",
    )
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "CSV file", type=["csv"],
            help="Required: timestamp, price. Optional: size, bid, ask, side.",
        )
    else:
        n_synth = st.slider("Synthetic events", 500, 10_000, 2_000, 500)
        synth_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    # ── FIB model ─────────────────────────────────────────────────────────────
    _section("FIB MODEL")
    model = st.selectbox(
        "Model", ["gaussian", "garch", "hawkes"],
        help="gaussian: N(μ,σ²) | garch: GARCH(1,1) volatility | hawkes: self-exciting point process",
    )
    info_mode = st.selectbox(
        "Information mode", ["observed", "expected"],
        help="observed: OPG (score outer-product) | expected: analytic Fisher info",
    )
    scalarizer = st.selectbox(
        "Scalarizer", ["logdet", "trace", "frobenius"],
        help="logdet: reparameterisation-invariant (recommended) | trace: fast | frobenius: norm",
    )

    st.divider()

    # ── Threshold ─────────────────────────────────────────────────────────────
    _section("ADAPTIVE THRESHOLD")
    eta = st.slider("η (eta) — threshold multiplier", 0.1, 10.0, 1.0, 0.1,
                    help="Higher → larger bars (more information per bar)")
    delta0 = st.slider("δ₀ — reference duration (s)", 1.0, 600.0, 60.0, 1.0,
                       help="Target bar duration at long-run information rate")
    ewma_alpha = st.slider("EWMA α", 0.01, 0.5, 0.05, 0.01,
                           help="Smaller = slower adaptation of threshold")
    min_warmup = st.number_input("Warmup bars", min_value=1, max_value=200, value=20,
                                 help="Bars before EWMA threshold is trusted")

    st.divider()

    # ── Timeout ───────────────────────────────────────────────────────────────
    _section("TIMEOUT / SAFETY VALVES")
    timeout_s = st.number_input("Bar timeout (s)", min_value=1.0, value=300.0, step=10.0,
                                help="Max wall-clock duration before force-close")
    use_inactivity = st.checkbox("Inactivity timeout", value=False)
    inactivity_s: Optional[float] = None
    if use_inactivity:
        inactivity_s = st.number_input("Inactivity gap (s)", min_value=1.0, value=60.0, step=5.0)
    max_events = st.number_input("Max events/bar", min_value=10, value=10_000, step=100)

    st.divider()

    # ── Model-specific ────────────────────────────────────────────────────────
    with st.expander("Model-specific parameters"):
        var_floor = st.number_input("Gaussian var floor", value=1e-12, format="%.2e",
                                    help="Minimum variance to prevent div-by-zero")
        garch_persist = st.slider("GARCH persistence max", 0.5, 0.9999, 0.9999, 0.0001)
        hawkes_floor = st.number_input("Hawkes intensity floor", value=1e-8, format="%.2e")
        eps_ridge = st.number_input("Ridge ε (regularization)", value=1e-6, format="%.2e",
                                    help="Added to J before scalarization for stability")

    st.divider()

    # ── Baseline comparison ───────────────────────────────────────────────────
    _section("BASELINE COMPARISON")
    show_baseline = st.checkbox("Show baseline bars", value=True)
    baseline_type = st.selectbox("Baseline type", ["time", "tick", "volume", "dollar"])
    if baseline_type == "time":
        baseline_param = st.number_input("Seconds/bar", min_value=1.0, value=60.0)
    elif baseline_type == "tick":
        baseline_param = st.number_input("Ticks/bar", min_value=2, value=50, step=5)
    elif baseline_type == "volume":
        baseline_param = st.number_input("Volume/bar", min_value=1.0, value=1000.0)
    else:
        baseline_param = st.number_input("Dollar/bar", min_value=1.0, value=100_000.0)

    st.divider()
    run_btn = st.button("▶  Build FIB Bars", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# Fisher Information Bars")
st.markdown(
    "A bar closes not when a clock ticks or a volume bucket fills, "
    "but when the accumulated **Fisher information** about the local "
    "market process reaches a target quantum *I\\**."
)

if not LIB_OK:
    st.error(f"Failed to import fibars library: {LIB_ERROR}")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
raw_df: Optional[pd.DataFrame] = None

if data_source == "Upload CSV" and uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif data_source == "Generate synthetic data":
    raw_df = _generate_synthetic(n=n_synth, seed=int(synth_seed))

if raw_df is None:
    st.info("👈  Upload a CSV or select **Generate synthetic data** in the sidebar, then click **Build FIB Bars**.")
    with st.expander("Expected CSV format"):
        st.markdown("""
| Column | Required | Type | Description |
|--------|----------|------|-------------|
| `timestamp` | ✅ | float / datetime | Unix seconds or ISO datetime |
| `price` | ✅ | float | Last trade or mid-quote |
| `size` | ☐ | float | Trade size |
| `bid` | ☐ | float | Best bid |
| `ask` | ☐ | float | Best ask |
| `side` | ☐ | str | 'B' or 'S' |
        """)
    st.stop()

# ── Column mapping ────────────────────────────────────────────────────────────
st.subheader("1 · Raw Data Preview")
cols_available = list(raw_df.columns)

with st.expander("Column mapping (expand if needed)", expanded=False):
    col1, col2, col3, col4, col5 = st.columns(5)
    ts_col   = col1.selectbox("timestamp", cols_available,
                               index=next((i for i, c in enumerate(cols_available) if "time" in c.lower()), 0))
    px_col   = col2.selectbox("price",     cols_available,
                               index=next((i for i, c in enumerate(cols_available) if "price" in c.lower() or "mid" in c.lower()), 0))
    sz_col   = col3.selectbox("size (opt)", ["(none)"] + cols_available)
    bid_col  = col4.selectbox("bid (opt)",  ["(none)"] + cols_available)
    ask_col  = col5.selectbox("ask (opt)",  ["(none)"] + cols_available)

# Build mapped dataframe
mapped: dict[str, pd.Series] = {"timestamp": raw_df[ts_col], "price": raw_df[px_col]}
if sz_col != "(none)":
    mapped["size"] = raw_df[sz_col]
if bid_col != "(none)":
    mapped["bid"] = raw_df[bid_col]
if ask_col != "(none)":
    mapped["ask"] = raw_df[ask_col]
mapped_df = pd.DataFrame(mapped)
mapped_df = _coerce_timestamp(mapped_df, "timestamp")

st.dataframe(mapped_df.head(200), use_container_width=True, height=200)
st.caption(f"{len(mapped_df):,} rows · {mapped_df.columns.tolist()}")

# ── Run ───────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Configure parameters in the sidebar and click **▶ Build FIB Bars**.")
    st.stop()

# Build config
try:
    cfg = FIBConfig(
        model=model,
        info_mode=info_mode,
        scalarizer=scalarizer,
        eta=eta,
        delta0_seconds=float(delta0),
        ewma_alpha=ewma_alpha,
        eps_ridge=float(eps_ridge),
        timeout_seconds=float(timeout_s),
        max_events_per_bar=int(max_events),
        inactivity_timeout_seconds=inactivity_s,
        min_warmup_events=int(min_warmup),
        var_floor=float(var_floor),
        garch_persistence_max=float(garch_persist),
        hawkes_intensity_floor=float(hawkes_floor),
    )
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# ── Build FIB bars ────────────────────────────────────────────────────────────
with st.spinner("Building FIB bars…"):
    try:
        bars_df = build_fib_bars(mapped_df, config=cfg)
    except Exception:
        st.error("FIB build failed:")
        st.code(traceback.format_exc())
        st.stop()

if bars_df.empty:
    st.warning("No bars were produced. Try adjusting timeout or eta.")
    st.stop()

aug_df = augment_with_fib_features(bars_df)
summary = _compute_summary(bars_df)

# ── Build baseline ────────────────────────────────────────────────────────────
baseline_df: Optional[pd.DataFrame] = None
if show_baseline:
    with st.spinner("Building baseline bars…"):
        try:
            kw: dict = {}
            if baseline_type == "time":   kw = {"seconds_per_bar": float(baseline_param)}
            elif baseline_type == "tick": kw = {"ticks_per_bar": int(baseline_param)}
            elif baseline_type == "volume": kw = {"volume_per_bar": float(baseline_param)}
            else:                           kw = {"dollar_per_bar": float(baseline_param)}
            baseline_df = build_baseline_bars(mapped_df, bar_type=baseline_type, **kw)
        except Exception:
            st.warning("Baseline build failed — skipping comparison.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 2 · Summary metrics
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("2 · Summary Metrics")

cols = st.columns(8)
metrics = [
    ("FIB Bars",         f"{summary['n_bars']:,}",                    ""),
    ("Avg Events/Bar",   f"{summary['avg_events_per_bar']:.1f}",      f"CV = {summary['cv_events']:.2f}"),
    ("Avg Duration",     f"{summary['avg_duration_s']:.1f}s",         f"med {summary['median_duration_s']:.1f}s"),
    ("% Threshold",      f"{summary['pct_threshold']:.1f}%",          "closed on I*"),
    ("Avg Info Scalar",  f"{summary['avg_info_scalar']:.3g}",         f"med {summary['median_info_scalar']:.3g}"),
    ("Model",            model.upper(),                                info_mode),
    ("Scalarizer",       scalarizer,                                   f"ε={eps_ridge:.0e}"),
    ("η / δ₀",           f"{eta} / {delta0}s",                       f"α={ewma_alpha}"),
]
for col, (lbl, val, sub) in zip(cols, metrics):
    col.markdown(_metric(lbl, val, sub), unsafe_allow_html=True)

# Baseline comparison row
if baseline_df is not None and not baseline_df.empty:
    bsummary = _compute_summary(baseline_df)
    st.markdown("**Baseline comparison:**")
    bcols = st.columns(4)
    bcols[0].metric("Baseline bars", f"{bsummary['n_bars']:,}",
                    delta=f"{bsummary['n_bars'] - summary['n_bars']:+,} vs FIB")
    bcols[1].metric("Avg events/bar", f"{bsummary['avg_events_per_bar']:.1f}",
                    delta=f"CV {bsummary['cv_events']:.2f} vs {summary['cv_events']:.2f}")
    bcols[2].metric("Avg duration (s)", f"{bsummary['avg_duration_s']:.1f}")
    bcols[3].metric("Bar type", f"{baseline_type} ({baseline_param:g})")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 3 · Bar tables
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("3 · FIB Bar Output")

tab1, tab2 = st.tabs(["FIB Bars", "Augmented Dataset"])

with tab1:
    st.dataframe(bars_df, use_container_width=True, height=300)

with tab2:
    st.dataframe(aug_df, use_container_width=True, height=300)
    st.caption("Augmented with derived FIB features for downstream ML / research.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 4 · Charts
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("4 · Charts")

# Chart 1: Price + bar boundaries
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=mapped_df["timestamp"], y=mapped_df["price"],
    mode="lines", name="Price", line=dict(color=C_BLUE, width=1),
))
# Bar open/close markers
fig_price.add_trace(go.Scatter(
    x=bars_df["open_time"], y=bars_df["open"],
    mode="markers", name="Bar open",
    marker=dict(color=C_GREEN, size=5, symbol="triangle-up"),
))
fig_price.add_trace(go.Scatter(
    x=bars_df["close_time"], y=bars_df["close"],
    mode="markers", name="Bar close",
    marker=dict(color=C_RED, size=5, symbol="triangle-down"),
))
fig_price.update_layout(
    template=PLOTLY_TEMPLATE, title="Price with FIB Bar Boundaries",
    xaxis_title="Time (s)", yaxis_title="Price",
    height=320, margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig_price, use_container_width=True)

# Charts row 2: duration, events/bar, info scalar, threshold utilization
c1, c2 = st.columns(2)

with c1:
    fig_dur = px.histogram(
        bars_df, x="duration_seconds", nbins=40,
        color="close_reason",
        color_discrete_map={
            "threshold": C_GREEN, "timeout": C_ORANGE,
            "flush": C_GREY, "inactivity": C_PURPLE, "max_events": C_RED,
        },
        title="Bar Duration Distribution",
        template=PLOTLY_TEMPLATE,
        labels={"duration_seconds": "Duration (s)", "close_reason": "Close reason"},
    )
    fig_dur.update_layout(height=300, margin=dict(l=40, r=10, t=40, b=40))
    st.plotly_chart(fig_dur, use_container_width=True)

with c2:
    fig_ne = px.histogram(
        bars_df, x="n_events", nbins=40,
        title="Events per Bar Distribution",
        template=PLOTLY_TEMPLATE,
        labels={"n_events": "Events per bar"},
    )
    fig_ne.update_traces(marker_color=C_BLUE)
    fig_ne.update_layout(height=300, margin=dict(l=40, r=10, t=40, b=40))
    st.plotly_chart(fig_ne, use_container_width=True)

# Chart 3: information scalar + threshold over bars
fig_info = make_subplots(specs=[[{"secondary_y": True}]])
fig_info.add_trace(go.Scatter(
    x=aug_df["bar_index"], y=aug_df["information_scalar"],
    name="Info scalar Φ(J)", line=dict(color=C_BLUE, width=1.5),
), secondary_y=False)
fig_info.add_trace(go.Scatter(
    x=aug_df["bar_index"], y=aug_df["threshold_at_close"],
    name="Threshold I*", line=dict(color=C_ORANGE, width=1.5, dash="dash"),
), secondary_y=False)
fig_info.add_trace(go.Bar(
    x=aug_df["bar_index"], y=aug_df["threshold_utilization"],
    name="Threshold utilization", marker_color=C_PURPLE, opacity=0.35,
), secondary_y=True)
fig_info.update_layout(
    template=PLOTLY_TEMPLATE, title="Information Scalar vs Threshold",
    xaxis_title="Bar index", height=320,
    margin=dict(l=50, r=50, t=40, b=40),
    legend=dict(orientation="h", y=-0.2),
)
fig_info.update_yaxes(title_text="Scalar / Threshold", secondary_y=False)
fig_info.update_yaxes(title_text="Utilization ratio", secondary_y=True)
st.plotly_chart(fig_info, use_container_width=True)

# Chart 4: information rate + close reason pie
c3, c4 = st.columns(2)

with c3:
    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(
        x=aug_df["bar_index"], y=aug_df["information_rate"],
        mode="lines+markers", name="Info rate",
        line=dict(color=C_GREEN, width=1.5),
        marker=dict(size=4),
    ))
    fig_rate.update_layout(
        template=PLOTLY_TEMPLATE, title="Information Rate (Φ/s)",
        xaxis_title="Bar index", yaxis_title="Info / second",
        height=280, margin=dict(l=50, r=10, t=40, b=40),
    )
    st.plotly_chart(fig_rate, use_container_width=True)

with c4:
    reason_counts = bars_df["close_reason"].value_counts()
    fig_pie = px.pie(
        values=reason_counts.values, names=reason_counts.index,
        color=reason_counts.index,
        color_discrete_map={
            "threshold": C_GREEN, "timeout": C_ORANGE,
            "flush": C_GREY, "inactivity": C_PURPLE, "max_events": C_RED,
        },
        title="Close Reason Distribution",
        template=PLOTLY_TEMPLATE, hole=0.4,
    )
    fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

# Chart 5: FIB vs Baseline comparison (if available)
if baseline_df is not None and not baseline_df.empty:
    st.markdown("**FIB vs Baseline Comparison**")
    c5, c6 = st.columns(2)

    with c5:
        compare_df = pd.DataFrame({
            "n_events": pd.concat([
                bars_df["n_events"].rename("FIB"),
                baseline_df["n_events"].rename("Baseline"),
            ], axis=1).melt(var_name="Type", value_name="n_events").dropna(),
        })
        fib_ne = bars_df["n_events"]
        bl_ne  = baseline_df["n_events"]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Histogram(
            x=fib_ne, name="FIB", opacity=0.65, marker_color=C_BLUE, nbinsx=30,
        ))
        fig_cmp.add_trace(go.Histogram(
            x=bl_ne, name=f"Baseline ({baseline_type})", opacity=0.65,
            marker_color=C_ORANGE, nbinsx=30,
        ))
        fig_cmp.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            title="Events/Bar: FIB vs Baseline",
            xaxis_title="Events", height=280,
            margin=dict(l=40, r=10, t=40, b=40),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with c6:
        fig_dur2 = go.Figure()
        fig_dur2.add_trace(go.Box(
            y=bars_df["duration_seconds"], name="FIB",
            marker_color=C_BLUE, boxmean=True,
        ))
        fig_dur2.add_trace(go.Box(
            y=baseline_df["duration_seconds"], name=f"Baseline ({baseline_type})",
            marker_color=C_ORANGE, boxmean=True,
        ))
        fig_dur2.update_layout(
            template=PLOTLY_TEMPLATE, title="Duration Distribution: FIB vs Baseline",
            yaxis_title="Duration (s)", height=280,
            margin=dict(l=40, r=10, t=40, b=40),
        )
        st.plotly_chart(fig_dur2, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# 5 · Downloads
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("5 · Downloads")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "⬇ FIB Bars CSV",
        data=_to_csv_bytes(bars_df),
        file_name="fib_bars.csv",
        mime="text/csv",
    )

with d2:
    st.download_button(
        "⬇ Augmented Dataset CSV",
        data=_to_csv_bytes(aug_df),
        file_name="fib_augmented.csv",
        mime="text/csv",
    )

with d3:
    summary_export = {k: v for k, v in summary.items() if k != "close_reasons"}
    summary_export.update(summary.get("close_reasons", {}))
    summary_export["config"] = {
        "model": model, "info_mode": info_mode, "scalarizer": scalarizer,
        "eta": eta, "delta0_seconds": delta0, "ewma_alpha": ewma_alpha,
        "timeout_seconds": float(timeout_s), "min_warmup_events": int(min_warmup),
    }
    st.download_button(
        "⬇ Summary JSON",
        data=json.dumps(summary_export, indent=2).encode("utf-8"),
        file_name="fib_summary.json",
        mime="application/json",
    )

with d4:
    if baseline_df is not None and not baseline_df.empty:
        st.download_button(
            "⬇ Baseline Bars CSV",
            data=_to_csv_bytes(baseline_df),
            file_name=f"baseline_{baseline_type}_bars.csv",
            mime="text/csv",
        )
    else:
        st.button("⬇ Baseline Bars CSV", disabled=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.75rem; margin-top:1rem;'>"
    "Fisher Information Bars · v1.1.0 · "
    "A bar closes when ∑ Ψᵢ(θ̂) ≥ I* — not when a clock ticks."
    "</div>",
    unsafe_allow_html=True,
)
