"""
Fisher Information Bars — Streamlit Demo  (self-contained)
============================================================
The entire fibars library is embedded so no pip install is required.
Run:  streamlit run app.py
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
import sys, types

def _bootstrap_fibars():
    """Inject fibars and all sub-modules into sys.modules at import time."""

    # ── utils.math ────────────────────────────────────────────────────────────
    import numpy as _np

    def symmetrize(A):
        return (A + A.T) * 0.5

    def ridge_logdet(J, eps=1e-6):
        n = J.shape[0]
        M = symmetrize(J) + eps * _np.eye(n)
        eigvals = _np.linalg.eigvalsh(M)
        floor = max(eps * 1e-3, 1e-300)
        eigvals = _np.maximum(eigvals, floor)
        return float(_np.sum(_np.log(eigvals)))

    def safe_outer(score):
        if not _np.all(_np.isfinite(score)):
            return _np.zeros((score.shape[0], score.shape[0]))
        return _np.outer(score, score)

    # ── events ────────────────────────────────────────────────────────────────
    from dataclasses import dataclass, field, asdict
    from typing import Optional, Literal

    @dataclass
    class MarketEvent:
        timestamp: float
        price: float
        size: float = 0.0
        bid: Optional[float] = None
        ask: Optional[float] = None
        side: Optional[str] = None
        event_type: Optional[str] = None
        index: int = field(default=0, repr=False)

        @property
        def spread(self):
            if self.bid is not None and self.ask is not None:
                return self.ask - self.bid
            return None

        @property
        def mid(self):
            if self.bid is not None and self.ask is not None:
                return (self.bid + self.ask) / 2.0
            return self.price

    # ── config ────────────────────────────────────────────────────────────────
    @dataclass
    class FIBConfig:
        model: str = "gaussian"
        info_mode: str = "observed"
        scalarizer: str = "logdet"
        eps_ridge: float = 1e-6
        eta: float = 1.0
        delta0_seconds: float = 60.0
        ewma_alpha: float = 0.05
        min_threshold: Optional[float] = None
        max_threshold: Optional[float] = None
        min_warmup_events: int = 20
        timeout_seconds: float = 300.0
        max_events_per_bar: int = 10_000
        inactivity_timeout_seconds: Optional[float] = None
        price_field: str = "price"
        volume_field: str = "size"
        var_floor: float = 1e-12
        garch_persistence_max: float = 0.9999
        hawkes_intensity_floor: float = 1e-8

        def __post_init__(self):
            if self.eta <= 0: raise ValueError(f"eta must be positive, got {self.eta}")
            if self.delta0_seconds <= 0: raise ValueError("delta0_seconds must be positive")
            if not (0 < self.ewma_alpha <= 1): raise ValueError(f"ewma_alpha must be in (0,1], got {self.ewma_alpha}")
            if self.eps_ridge < 0: raise ValueError("eps_ridge must be non-negative")
            if self.model not in ("gaussian","garch","hawkes"): raise ValueError(f"Unknown model '{self.model}'")
            if self.info_mode not in ("observed","expected"): raise ValueError(f"Unknown info_mode '{self.info_mode}'")
            if self.scalarizer not in ("logdet","trace","frobenius"): raise ValueError(f"Unknown scalarizer '{self.scalarizer}'")
            if self.timeout_seconds <= 0: raise ValueError("timeout_seconds must be positive")
            if self.max_events_per_bar < 1: raise ValueError("max_events_per_bar must be >= 1")
            if self.inactivity_timeout_seconds is not None and self.inactivity_timeout_seconds <= 0:
                raise ValueError("inactivity_timeout_seconds must be positive")
            if self.min_threshold is not None and self.max_threshold is not None:
                if self.min_threshold >= self.max_threshold: raise ValueError("min_threshold must be < max_threshold")
            if self.var_floor <= 0: raise ValueError("var_floor must be positive")
            if not (0 < self.garch_persistence_max < 1): raise ValueError("garch_persistence_max must be in (0,1)")
            if self.hawkes_intensity_floor <= 0: raise ValueError("hawkes_intensity_floor must be positive")

    # ── models ────────────────────────────────────────────────────────────────
    class GaussianModel:
        n_params = 2
        def __init__(self, var_floor=1e-12):
            self._var_floor = max(var_floor, 1e-300)
        def initialize(self):
            self._mu = 0.0; self._sigma2 = 1.0; self._n = 0; self._M2 = 0.0
        def update(self, event):
            x = event.price; self._n += 1
            delta = x - self._mu; self._mu += delta / self._n
            delta2 = x - self._mu; self._M2 += delta * delta2
            if self._n >= 2:
                self._sigma2 = max(self._M2 / (self._n - 1), self._var_floor)
        def score(self, event):
            if self._n == 0: return _np.zeros(2)
            s2 = max(self._sigma2, self._var_floor); r = event.price - self._mu
            return _np.array([r / s2, -0.5 / s2 + 0.5 * r**2 / s2**2])
        def observed_information_increment(self, event):
            s = self.score(event)
            return _np.zeros((2,2)) if not _np.all(_np.isfinite(s)) else _np.outer(s, s)
        def expected_information_increment(self, event):
            if self._n == 0: return _np.zeros((2,2))
            s2 = max(self._sigma2, self._var_floor)
            I = _np.zeros((2,2)); I[0,0] = 1.0/s2; I[1,1] = 0.5/s2**2; return I
        def state_dict(self): return {"mu":self._mu,"sigma2":self._sigma2,"n":self._n}

    class GARCHModel:
        n_params = 3
        def __init__(self, omega_init=1e-6, alpha_init=0.05, beta_init=0.90,
                     persistence_max=0.9999, var_floor=1e-12):
            total = alpha_init + beta_init
            if total >= persistence_max:
                scale = (persistence_max - 1e-9) / total
                alpha_init *= scale; beta_init *= scale
            self._omega_init = max(omega_init, 1e-12)
            self._alpha_init = alpha_init; self._beta_init = beta_init
            self._persistence_max = persistence_max
            self._var_floor = max(var_floor, 1e-300)
        def initialize(self):
            self._omega = self._omega_init; self._alpha = self._alpha_init
            self._beta = self._beta_init
            self._sigma2 = self._omega_init / max(1.0 - self._alpha_init - self._beta_init, 1e-6)
            self._prev_eps2 = 0.0; self._prev_price = None; self._n = 0
        def update(self, event):
            if self._prev_price is not None:
                eps = event.price - self._prev_price; self._prev_eps2 = eps * eps
                self._sigma2 = max(self._omega + self._alpha*self._prev_eps2 + self._beta*self._sigma2, self._var_floor)
            self._prev_price = event.price; self._n += 1
        def score(self, event):
            if self._prev_price is None: return _np.zeros(3)
            eps = event.price - self._prev_price; eps2 = eps*eps
            s2 = max(self._sigma2, self._var_floor); u = eps2/s2 - 1.0; f = u/(2.0*s2)
            return _np.array([f, f*self._prev_eps2, f*self._sigma2])
        def observed_information_increment(self, event):
            s = self.score(event)
            return _np.zeros((3,3)) if not _np.all(_np.isfinite(s)) else _np.outer(s,s)
        def expected_information_increment(self, event):
            return self.observed_information_increment(event)
        def state_dict(self): return {"omega":self._omega,"alpha":self._alpha,"beta":self._beta,"sigma2":self._sigma2,"n":self._n}

    class HawkesModel:
        n_params = 3
        def __init__(self, mu_init=0.1, alpha_init=0.5, beta_init=1.0, intensity_floor=1e-8):
            self._mu_init = max(mu_init, 1e-10); self._alpha_init = max(alpha_init, 0.0)
            self._beta_init = max(beta_init, 1e-10); self._intensity_floor = max(intensity_floor, 1e-300)
        def initialize(self):
            self._mu = self._mu_init; self._alpha = self._alpha_init
            self._beta = self._beta_init; self._R = 0.0; self._prev_time = None; self._n = 0
        def _intensity(self, R):
            return max(self._mu + self._alpha * R, self._intensity_floor)
        def _decay(self, dt):
            return float(_np.exp(-self._beta * max(dt, 0.0)))
        def update(self, event):
            if self._prev_time is not None:
                dt = event.timestamp - self._prev_time
                self._R = self._decay(dt) * (self._R + 1.0)
            self._prev_time = event.timestamp; self._n += 1
        def score(self, event):
            if self._prev_time is None: return _np.zeros(3)
            dt = event.timestamp - self._prev_time
            decay = self._decay(dt); R_new = decay * (self._R + 1.0)
            lam = self._intensity(R_new); inv = 1.0 / lam
            s = _np.array([inv, inv * R_new, -inv * self._alpha * R_new * dt])
            return s if _np.all(_np.isfinite(s)) else _np.zeros(3)
        def observed_information_increment(self, event):
            s = self.score(event)
            return _np.zeros((3,3)) if not _np.all(_np.isfinite(s)) else _np.outer(s,s)
        def expected_information_increment(self, event):
            return self.observed_information_increment(event)
        def state_dict(self): return {"mu":self._mu,"alpha":self._alpha,"beta":self._beta,"R":self._R,"n":self._n}

    def _create_model(cfg):
        if cfg.model == "gaussian":
            m = GaussianModel(var_floor=cfg.var_floor)
        elif cfg.model == "garch":
            m = GARCHModel(persistence_max=cfg.garch_persistence_max, var_floor=cfg.var_floor)
        elif cfg.model == "hawkes":
            m = HawkesModel(intensity_floor=cfg.hawkes_intensity_floor)
        else:
            raise ValueError(f"Unknown model '{cfg.model}'")
        m.initialize()
        return m

    # ── scalarizers ───────────────────────────────────────────────────────────
    class LogdetScalarizer:
        def __call__(self, J, eps=1e-6):
            return ridge_logdet(symmetrize(J), eps=eps)
    class TraceScalarizer:
        def __call__(self, J, eps=1e-6):
            return float(_np.trace(J) + eps * J.shape[0])
    class FrobeniusScalarizer:
        def __call__(self, J, eps=1e-6):
            return max(float(_np.linalg.norm(J, "fro")), eps)

    _SCALARIZERS = {"logdet": LogdetScalarizer(), "trace": TraceScalarizer(), "frobenius": FrobeniusScalarizer()}
    def _get_scalarizer(name):
        s = _SCALARIZERS.get(name)
        if s is None: raise ValueError(f"Unknown scalarizer '{name}'")
        return s

    # ── adaptive threshold ────────────────────────────────────────────────────
    class AdaptiveThresholdPolicy:
        def __init__(self, eta=1.0, delta0_seconds=60.0, ewma_alpha=0.05,
                     min_threshold=None, max_threshold=None, min_warmup_events=20):
            self._eta=eta; self._delta0=delta0_seconds; self._alpha=ewma_alpha
            self._min=min_threshold; self._max=max_threshold; self._warmup=min_warmup_events
            self._n=0; self._ewma=None; self._last=0.0; self._thr=float("inf"); self._seeded=False
        def current_threshold(self): return self._thr
        def _clamp(self, v):
            if self._min is not None: v=max(v,self._min)
            if self._max is not None: v=min(v,self._max)
            return v
        def seed_from_scalar(self, s):
            if not self._seeded and s > 0.0:
                self._last=s; self._thr=self._clamp(s); self._seeded=True
        def update(self, scalar, dur):
            self._last=scalar; self._n+=1; rate=scalar/max(dur,1e-3)
            self._ewma=rate if self._ewma is None else self._alpha*rate+(1-self._alpha)*self._ewma
            if self._n>=self._warmup and self._ewma is not None:
                self._thr=self._clamp(self._eta*self._ewma*self._delta0)
            else:
                self._thr=self._clamp(max(self._last,1e-9))
        @property
        def n_bars_completed(self): return self._n
        def state_dict(self): return {"n_bars_completed":self._n,"ewma_rate":self._ewma,"threshold":self._thr}

    # ── aggregator ────────────────────────────────────────────────────────────
    class BarAggregator:
        def __init__(self, price_field="price", volume_field="size"):
            self._pf=price_field; self._vf=volume_field; self.reset()
        def reset(self):
            self._n=0; self._open=None; self._high=float("-inf"); self._low=float("inf")
            self._close=None; self._vol=0.0; self._dollar=0.0
            self._ss=0.0; self._sc=0; self._ot=None; self._ct=None; self._si=0; self._ei=0
        def add(self, ev):
            p=float(getattr(ev,self._pf,ev.price)); s=float(getattr(ev,self._vf,ev.size))
            if self._n==0: self._open=p; self._ot=ev.timestamp; self._si=ev.index
            self._high=max(self._high,p); self._low=min(self._low,p)
            self._close=p; self._ct=ev.timestamp; self._ei=ev.index
            self._vol+=s; self._dollar+=p*s
            sp=ev.spread
            if sp is not None and sp>=0: self._ss+=sp; self._sc+=1
            self._n+=1
        @property
        def n_events(self): return self._n
        @property
        def open(self): return self._open or 0.0
        @property
        def high(self): return self._high if self._n>0 else 0.0
        @property
        def low(self): return self._low if self._n>0 else 0.0
        @property
        def close(self): return self._close or 0.0
        @property
        def sum_volume(self): return self._vol
        @property
        def dollar_value(self): return self._dollar
        @property
        def mean_spread(self): return (self._ss/self._sc) if self._sc>0 else None
        @property
        def open_time(self): return self._ot
        @property
        def close_time(self): return self._ct
        @property
        def duration_seconds(self):
            return max(self._ct-self._ot,0.0) if self._ot is not None and self._ct is not None else 0.0
        @property
        def start_index(self): return self._si
        @property
        def end_index(self): return self._ei

    # ── FIBBar ────────────────────────────────────────────────────────────────
    import pandas as _pd

    @dataclass
    class FIBBar:
        open_time: float; close_time: float; duration_seconds: float; n_events: int
        open: float; high: float; low: float; close: float
        sum_volume: float; dollar_value: float; mean_spread: Optional[float]
        information_scalar: float; threshold_at_close: float
        timeout_flag: bool; close_reason: str
        model_name: str; info_mode: str; scalarizer_name: str
        start_event_index: int; end_event_index: int

        def to_dict(self): return asdict(self)

        @classmethod
        def to_dataframe(cls, bars):
            return _pd.DataFrame([b.to_dict() for b in bars]) if bars else _pd.DataFrame()

    # ── FIBBuilder ────────────────────────────────────────────────────────────
    class FIBBuilder:
        def __init__(self, config=None):
            self.cfg = config or FIBConfig()
            self._model = _create_model(self.cfg)
            self._sc = _get_scalarizer(self.cfg.scalarizer)
            self._tp = AdaptiveThresholdPolicy(
                eta=self.cfg.eta, delta0_seconds=self.cfg.delta0_seconds,
                ewma_alpha=self.cfg.ewma_alpha, min_threshold=self.cfg.min_threshold,
                max_threshold=self.cfg.max_threshold, min_warmup_events=self.cfg.min_warmup_events,
            )
            self._agg = BarAggregator(self.cfg.price_field, self.cfg.volume_field)
            self._np = self._model.n_params
            self._J = _np.zeros((self._np, self._np))
            self._cur = 0.0; self._gec = 0; self._let = 0.0

        def _reset(self):
            self._J = _np.zeros((self._np, self._np)); self._cur = 0.0; self._agg.reset()

        def _inc(self, ev):
            inc = (self._model.observed_information_increment(ev)
                   if self.cfg.info_mode == "observed"
                   else self._model.expected_information_increment(ev))
            return inc if _np.all(_np.isfinite(inc)) else _np.zeros((self._np, self._np))

        def _scalar(self):
            v = self._sc(symmetrize(self._J), eps=self.cfg.eps_ridge)
            return v if _np.isfinite(v) else 0.0

        def _emit(self, reason):
            s = self._cur; thr = self._tp.current_threshold(); dur = self._agg.duration_seconds
            self._tp.update(s, max(dur, 1e-3))
            bar = FIBBar(
                open_time=self._agg.open_time or 0.0, close_time=self._agg.close_time or 0.0,
                duration_seconds=dur, n_events=self._agg.n_events,
                open=self._agg.open, high=self._agg.high, low=self._agg.low, close=self._agg.close,
                sum_volume=self._agg.sum_volume, dollar_value=self._agg.dollar_value,
                mean_spread=self._agg.mean_spread, information_scalar=s,
                threshold_at_close=thr, timeout_flag=(reason != "threshold"),
                close_reason=reason, model_name=self.cfg.model, info_mode=self.cfg.info_mode,
                scalarizer_name=self.cfg.scalarizer,
                start_event_index=self._agg.start_index, end_event_index=self._agg.end_index,
            )
            self._reset(); return bar

        def update(self, ev):
            self._gec += 1; ev.index = self._gec
            prev_t = self._let; self._let = ev.timestamp
            if self._agg.n_events == 0:
                self._agg.add(ev); self._model.update(ev); return None
            self._J += self._inc(ev); self._cur = self._scalar()
            self._tp.seed_from_scalar(self._cur)
            self._agg.add(ev); self._model.update(ev)
            bs = self._agg.open_time or ev.timestamp
            if self.cfg.inactivity_timeout_seconds is not None and prev_t > 0:
                if ev.timestamp - prev_t >= self.cfg.inactivity_timeout_seconds:
                    return self._emit("inactivity")
            if ev.timestamp - bs >= self.cfg.timeout_seconds:
                return self._emit("timeout")
            if self._agg.n_events >= self.cfg.max_events_per_bar:
                return self._emit("max_events")
            if self._cur >= self._tp.current_threshold():
                return self._emit("threshold")
            return None

        def flush(self):
            return self._emit("flush") if self._agg.n_events > 0 else None

        @property
        def current_scalar(self): return self._cur
        @property
        def current_threshold(self): return self._tp.current_threshold()
        @property
        def n_bars_completed(self): return self._tp.n_bars_completed

    # ── baseline builders ─────────────────────────────────────────────────────
    def _make_base_bar(agg, btype, reason):
        return FIBBar(
            open_time=agg.open_time or 0.0, close_time=agg.close_time or 0.0,
            duration_seconds=agg.duration_seconds, n_events=agg.n_events,
            open=agg.open, high=agg.high, low=agg.low, close=agg.close,
            sum_volume=agg.sum_volume, dollar_value=agg.dollar_value, mean_spread=agg.mean_spread,
            information_scalar=0.0, threshold_at_close=0.0,
            timeout_flag=(reason != "threshold"), close_reason=reason,
            model_name="baseline", info_mode="none", scalarizer_name=btype,
            start_event_index=agg.start_index, end_event_index=agg.end_index,
        )

    def _build_time_bars(events, spb=60.0):
        bars=[]; agg=BarAggregator(); end=None
        for i,ev in enumerate(events):
            ev.index=i+1
            if agg.n_events==0: agg.add(ev); end=ev.timestamp+spb; continue
            if ev.timestamp>=end:
                bars.append(_make_base_bar(agg,"time","threshold")); agg.reset(); end=None
            agg.add(ev)
            if end is None: end=ev.timestamp+spb
        if agg.n_events>0: bars.append(_make_base_bar(agg,"time","flush"))
        return bars

    def _build_tick_bars(events, tpb=100):
        bars=[]; agg=BarAggregator()
        for i,ev in enumerate(events):
            ev.index=i+1; agg.add(ev)
            if agg.n_events>=tpb: bars.append(_make_base_bar(agg,"tick","threshold")); agg.reset()
        if agg.n_events>0: bars.append(_make_base_bar(agg,"tick","flush"))
        return bars

    def _build_volume_bars(events, vpb=1000.0):
        bars=[]; agg=BarAggregator(); cv=0.0
        for i,ev in enumerate(events):
            ev.index=i+1; agg.add(ev); cv+=ev.size
            if cv>=vpb: bars.append(_make_base_bar(agg,"volume","threshold")); agg.reset(); cv=0.0
        if agg.n_events>0: bars.append(_make_base_bar(agg,"volume","flush"))
        return bars

    def _build_dollar_bars(events, dpb=100_000.0):
        bars=[]; agg=BarAggregator(); cd=0.0
        for i,ev in enumerate(events):
            ev.index=i+1; agg.add(ev); cd+=ev.price*ev.size
            if cd>=dpb: bars.append(_make_base_bar(agg,"dollar","threshold")); agg.reset(); cd=0.0
        if agg.n_events>0: bars.append(_make_base_bar(agg,"dollar","flush"))
        return bars

    # ── adapters ──────────────────────────────────────────────────────────────
    import numpy as _np2

    def _isnan(v):
        try: return _np2.isnan(float(v))
        except: return v is None

    def _df_to_events(df):
        missing = [c for c in ("timestamp","price") if c not in df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")
        if df.empty: return []
        if df["price"].isnull().any(): raise ValueError("'price' contains NaN")
        if df["timestamp"].isnull().any(): raise ValueError("'timestamp' contains NaN")
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp").reset_index(drop=True)
        hb="bid" in df.columns; ha="ask" in df.columns
        hs="size" in df.columns; hsi="side" in df.columns; he="event_type" in df.columns
        evs=[]
        for row in df.itertuples(index=False):
            evs.append(MarketEvent(
                timestamp=float(row.timestamp), price=float(row.price),
                size=float(row.size) if hs else 0.0,
                bid=float(row.bid) if hb and not _isnan(row.bid) else None,
                ask=float(row.ask) if ha and not _isnan(row.ask) else None,
                side=str(row.side) if hsi and row.side is not None else None,
                event_type=str(row.event_type) if he and row.event_type is not None else None,
            ))
        return evs

    # ── public API ────────────────────────────────────────────────────────────
    def build_fib_bars(data, config=None, **kw):
        cfg = config or FIBConfig(**{k:v for k,v in kw.items()
                                     if k in FIBConfig.__dataclass_fields__})
        events = _df_to_events(data)
        builder = FIBBuilder(cfg); bars=[]
        for ev in events:
            b=builder.update(ev)
            if b: bars.append(b)
        b=builder.flush()
        if b: bars.append(b)
        return FIBBar.to_dataframe(bars)

    def build_baseline_bars(data, bar_type="time", seconds_per_bar=60.0,
                            ticks_per_bar=100, volume_per_bar=1000.0, dollar_per_bar=100_000.0):
        events = _df_to_events(data)
        if bar_type=="time":   bars=_build_time_bars(events, seconds_per_bar)
        elif bar_type=="tick": bars=_build_tick_bars(events, ticks_per_bar)
        elif bar_type=="volume": bars=_build_volume_bars(events, volume_per_bar)
        elif bar_type=="dollar": bars=_build_dollar_bars(events, dollar_per_bar)
        else: raise ValueError(f"Unknown bar_type '{bar_type}'")
        return FIBBar.to_dataframe(bars)

    def augment_with_fib_features(df):
        if df.empty: return df.copy()
        import numpy as np
        out = df.copy()
        thr = out["threshold_at_close"].clip(lower=1e-12)
        out["threshold_utilization"] = out["information_scalar"] / thr
        dur = out["duration_seconds"].clip(lower=1e-3)
        out["information_rate"] = out["information_scalar"] / dur
        out["log_duration"] = np.log1p(out["duration_seconds"])
        out["log_n_events"] = np.log1p(out["n_events"])
        out["log_information_scalar"] = np.log1p(out["information_scalar"].clip(lower=0))
        out["price_range"] = out["high"] - out["low"]
        op = out["open"].replace(0, float("nan"))
        out["price_range_pct"] = (out["price_range"] / op * 100).fillna(0.0)
        out["vwap"] = out["dollar_value"] / out["sum_volume"].clip(lower=1e-12)
        out["is_threshold_close"] = out["close_reason"] == "threshold"
        out["bar_index"] = np.arange(len(out))
        return out

    # Expose at module level
    return {
        "FIBConfig": FIBConfig,
        "MarketEvent": MarketEvent,
        "FIBBar": FIBBar,
        "FIBBuilder": FIBBuilder,
        "build_fib_bars": build_fib_bars,
        "build_baseline_bars": build_baseline_bars,
        "augment_with_fib_features": augment_with_fib_features,
    }

_FIBARS = _bootstrap_fibars()
FIBConfig              = _FIBARS["FIBConfig"]
MarketEvent            = _FIBARS["MarketEvent"]
FIBBar                 = _FIBARS["FIBBar"]
FIBBuilder             = _FIBARS["FIBBuilder"]
build_fib_bars         = _FIBARS["build_fib_bars"]
build_baseline_bars    = _FIBARS["build_baseline_bars"]
augment_with_fib_features = _FIBARS["augment_with_fib_features"]

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════
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

st.set_page_config(
    page_title="Fisher Information Bars",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container{padding-top:1.5rem;padding-bottom:1rem}
    .metric-card{background:#0e1117;border:1px solid #2a2d3e;border-radius:8px;
                 padding:1rem 1.2rem;text-align:center}
    .metric-label{font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.06em}
    .metric-value{font-size:1.6rem;font-weight:700;color:#e8e8e8;margin-top:.2rem}
    .metric-sub{font-size:.75rem;color:#aaa;margin-top:.1rem}
    .section-header{font-size:.75rem;font-weight:600;letter-spacing:.1em;
        text-transform:uppercase;color:#aaa;border-bottom:1px solid #2a2d3e;
        padding-bottom:.3rem;margin-bottom:.8rem}
    div[data-testid="stExpander"]{border:1px solid #2a2d3e!important;border-radius:6px}
    .stDownloadButton>button{width:100%;border-radius:6px;background:#1a1d2e;
        border:1px solid #3a3d5e;color:#ccc}
    .stDownloadButton>button:hover{background:#2a2d4e;color:#fff}
</style>
""", unsafe_allow_html=True)

C_BLUE="#4e8ef7"; C_ORANGE="#f79e4e"; C_GREEN="#4ef7a0"
C_RED="#f74e4e"; C_PURPLE="#b44ef7"; C_GREY="#666"
PT = "plotly_dark"

def _metric(label, value, sub=""):
    sub_h = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{sub_h}</div>')

def _section(t): st.markdown(f'<div class="section-header">{t}</div>', unsafe_allow_html=True)

def _csv(df): return df.to_csv(index=False).encode("utf-8")

def _coerce_ts(df, col):
    df = df.copy(); s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        df[col] = s.astype(float); return df
    try:
        df[col] = pd.to_datetime(s, utc=True).astype("int64") / 1e9; return df
    except Exception:
        pass
    try: df[col] = s.astype(float)
    except Exception: pass
    return df

def _summary(df):
    if df.empty: return {}
    n=len(df); ne=df["n_events"]; dur=df["duration_seconds"]; info=df["information_scalar"]
    thr=(df["close_reason"]=="threshold").sum()
    cv=ne.std()/ne.mean() if ne.mean()>0 else float("nan")
    return {
        "n_bars":n, "avg_events":float(ne.mean()), "cv_events":float(cv),
        "avg_dur":float(dur.mean()), "med_dur":float(dur.median()),
        "pct_thr":float(thr/n*100), "avg_info":float(info.mean()),
        "med_info":float(info.median()),
        "reasons":df["close_reason"].value_counts().to_dict(),
    }

def _synthetic(n=2000, seed=42):
    rng=np.random.default_rng(seed); price=100.0
    prices,ts,sizes=[],[],[]
    t=0.0
    for _ in range(n):
        vol=rng.choice([0.05,0.3,0.8],p=[0.7,0.2,0.1])
        price=max(price+rng.normal(0,vol),1.0)
        t+=rng.exponential(0.5)
        prices.append(round(price,4)); ts.append(round(t,4))
        sizes.append(int(rng.integers(1,20)))
    df=pd.DataFrame({"timestamp":ts,"price":prices,"size":sizes})
    h=0.02+0.01*rng.random(n); df["bid"]=df["price"]-h; df["ask"]=df["price"]+h
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Fisher Information Bars")
    st.markdown("*Information-geometric financial sampling*")
    st.divider()

    _section("DATA SOURCE")
    data_source = st.radio("Source", ["Upload CSV","Generate synthetic data"],
                            label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("CSV file", type=["csv"],
            help="Required: timestamp, price. Optional: size, bid, ask.")
    else:
        n_synth = st.slider("Synthetic events", 500, 10_000, 2_000, 500)
        synth_seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    _section("FIB MODEL")
    model      = st.selectbox("Model", ["gaussian","garch","hawkes"],
        help="gaussian: N(μ,σ²) | garch: GARCH(1,1) | hawkes: self-exciting")
    info_mode  = st.selectbox("Information mode", ["observed","expected"],
        help="observed: OPG | expected: analytic Fisher info")
    scalarizer = st.selectbox("Scalarizer", ["logdet","trace","frobenius"],
        help="logdet: reparameterisation-invariant (recommended)")

    st.divider()
    _section("ADAPTIVE THRESHOLD")
    eta       = st.slider("η — threshold multiplier", 0.1, 10.0, 1.0, 0.1)
    delta0    = st.slider("δ₀ — reference duration (s)", 1.0, 600.0, 60.0, 1.0)
    ewma_a    = st.slider("EWMA α", 0.01, 0.5, 0.05, 0.01)
    min_wu    = st.number_input("Warmup bars", min_value=1, max_value=200, value=20)

    st.divider()
    _section("TIMEOUT / SAFETY VALVES")
    timeout_s  = st.number_input("Bar timeout (s)", min_value=1.0, value=300.0, step=10.0)
    use_inact  = st.checkbox("Inactivity timeout", value=False)
    inact_s: Optional[float] = None
    if use_inact:
        inact_s = st.number_input("Inactivity gap (s)", min_value=1.0, value=60.0, step=5.0)
    max_ev = st.number_input("Max events/bar", min_value=10, value=10_000, step=100)

    st.divider()
    with st.expander("Model-specific parameters"):
        var_floor   = st.number_input("Gaussian var floor",  value=1e-12, format="%.2e")
        garch_p     = st.slider("GARCH persistence max", 0.5, 0.9999, 0.9999, 0.0001)
        hawkes_fl   = st.number_input("Hawkes intensity floor", value=1e-8, format="%.2e")
        eps_ridge   = st.number_input("Ridge ε", value=1e-6, format="%.2e")

    st.divider()
    _section("BASELINE COMPARISON")
    show_bl  = st.checkbox("Show baseline bars", value=True)
    bl_type  = st.selectbox("Baseline type", ["time","tick","volume","dollar"])
    if bl_type=="time":   bl_p = st.number_input("Seconds/bar", min_value=1.0, value=60.0)
    elif bl_type=="tick": bl_p = st.number_input("Ticks/bar", min_value=2, value=50, step=5)
    elif bl_type=="volume": bl_p = st.number_input("Volume/bar", min_value=1.0, value=1000.0)
    else: bl_p = st.number_input("Dollar/bar", min_value=1.0, value=100_000.0)

    st.divider()
    run_btn = st.button("▶  Build FIB Bars", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Fisher Information Bars")
st.markdown(
    "A bar closes not when a clock ticks or a volume bucket fills, "
    "but when the accumulated **Fisher information** reaches target quantum *I\\**."
)

raw_df = None
if data_source=="Upload CSV" and uploaded_file is not None:
    try: raw_df = pd.read_csv(uploaded_file)
    except Exception as e: st.error(f"Failed to read CSV: {e}"); st.stop()
elif data_source=="Generate synthetic data":
    raw_df = _synthetic(n=n_synth, seed=int(synth_seed))

if raw_df is None:
    st.info("👈  Upload a CSV or select **Generate synthetic data**, then click **▶ Build FIB Bars**.")
    with st.expander("Expected CSV format"):
        st.markdown("""
| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | ✅ | Unix seconds or ISO datetime |
| `price` | ✅ | Last trade or mid-quote |
| `size` | ☐ | Trade size |
| `bid` / `ask` | ☐ | Best bid/ask |
        """)
    st.stop()

# ── Column mapping ────────────────────────────────────────────────────────────
st.subheader("1 · Raw Data Preview")
cols = list(raw_df.columns)
with st.expander("Column mapping", expanded=False):
    c1,c2,c3,c4,c5 = st.columns(5)
    ts_col  = c1.selectbox("timestamp", cols, index=next((i for i,c in enumerate(cols) if "time" in c.lower()),0))
    px_col  = c2.selectbox("price",     cols, index=next((i for i,c in enumerate(cols) if "price" in c.lower() or "mid" in c.lower()),0))
    sz_col  = c3.selectbox("size (opt)", ["(none)"]+cols)
    bid_col = c4.selectbox("bid (opt)",  ["(none)"]+cols)
    ask_col = c5.selectbox("ask (opt)",  ["(none)"]+cols)

mapped: dict = {"timestamp": raw_df[ts_col], "price": raw_df[px_col]}
if sz_col  != "(none)": mapped["size"] = raw_df[sz_col]
if bid_col != "(none)": mapped["bid"]  = raw_df[bid_col]
if ask_col != "(none)": mapped["ask"]  = raw_df[ask_col]
mapped_df = _coerce_ts(pd.DataFrame(mapped), "timestamp")

st.dataframe(mapped_df.head(200), use_container_width=True, height=200)
st.caption(f"{len(mapped_df):,} rows · {list(mapped_df.columns)}")

if not run_btn:
    st.info("Configure parameters in the sidebar and click **▶ Build FIB Bars**.")
    st.stop()

# ── Build ─────────────────────────────────────────────────────────────────────
try:
    cfg = FIBConfig(
        model=model, info_mode=info_mode, scalarizer=scalarizer,
        eta=eta, delta0_seconds=float(delta0), ewma_alpha=ewma_a,
        eps_ridge=float(eps_ridge), timeout_seconds=float(timeout_s),
        max_events_per_bar=int(max_ev), inactivity_timeout_seconds=inact_s,
        min_warmup_events=int(min_wu), var_floor=float(var_floor),
        garch_persistence_max=float(garch_p), hawkes_intensity_floor=float(hawkes_fl),
    )
except ValueError as e:
    st.error(f"Configuration error: {e}"); st.stop()

with st.spinner("Building FIB bars…"):
    try: bars_df = build_fib_bars(mapped_df, config=cfg)
    except Exception: st.error("FIB build failed:"); st.code(traceback.format_exc()); st.stop()

if bars_df.empty: st.warning("No bars produced. Try reducing timeout or eta."); st.stop()

aug_df = augment_with_fib_features(bars_df)
sm = _summary(bars_df)

bl_df = None
if show_bl:
    with st.spinner("Building baseline bars…"):
        try:
            kw = {}
            if bl_type=="time":   kw={"seconds_per_bar":float(bl_p)}
            elif bl_type=="tick": kw={"ticks_per_bar":int(bl_p)}
            elif bl_type=="volume": kw={"volume_per_bar":float(bl_p)}
            else: kw={"dollar_per_bar":float(bl_p)}
            bl_df = build_baseline_bars(mapped_df, bar_type=bl_type, **kw)
        except Exception: st.warning("Baseline build failed — skipping.")

st.divider()

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("2 · Summary Metrics")
mcols = st.columns(8)
metrics = [
    ("FIB Bars",        f"{sm['n_bars']:,}",               ""),
    ("Avg Events/Bar",  f"{sm['avg_events']:.1f}",         f"CV={sm['cv_events']:.2f}"),
    ("Avg Duration",    f"{sm['avg_dur']:.1f}s",           f"med {sm['med_dur']:.1f}s"),
    ("% Threshold",     f"{sm['pct_thr']:.1f}%",           "closed on I*"),
    ("Avg Info Scalar", f"{sm['avg_info']:.3g}",           f"med {sm['med_info']:.3g}"),
    ("Model",           model.upper(),                      info_mode),
    ("Scalarizer",      scalarizer,                         f"ε={eps_ridge:.0e}"),
    ("η / δ₀",          f"{eta} / {delta0}s",              f"α={ewma_a}"),
]
for col, (lbl,val,sub) in zip(mcols, metrics):
    col.markdown(_metric(lbl, val, sub), unsafe_allow_html=True)

if bl_df is not None and not bl_df.empty:
    bsm = _summary(bl_df)
    st.markdown("**Baseline comparison:**")
    bc = st.columns(4)
    bc[0].metric("Baseline bars", f"{bsm['n_bars']:,}", delta=f"{bsm['n_bars']-sm['n_bars']:+,} vs FIB")
    bc[1].metric("Avg events/bar", f"{bsm['avg_events']:.1f}", delta=f"CV {bsm['cv_events']:.2f} vs {sm['cv_events']:.2f}")
    bc[2].metric("Avg duration (s)", f"{bsm['avg_dur']:.1f}")
    bc[3].metric("Bar type", f"{bl_type} ({bl_p:g})")

st.divider()

# ── Tables ────────────────────────────────────────────────────────────────────
st.subheader("3 · Bar Output")
t1, t2 = st.tabs(["FIB Bars", "Augmented Dataset"])
with t1: st.dataframe(bars_df, use_container_width=True, height=280)
with t2:
    st.dataframe(aug_df, use_container_width=True, height=280)
    st.caption("Augmented with derived features: threshold utilization, info rate, log-transforms, VWAP, …")
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("4 · Charts")

# Price + bar boundaries
fig_px = go.Figure()
fig_px.add_trace(go.Scatter(x=mapped_df["timestamp"],y=mapped_df["price"],
    mode="lines",name="Price",line=dict(color=C_BLUE,width=1)))
fig_px.add_trace(go.Scatter(x=bars_df["open_time"],y=bars_df["open"],
    mode="markers",name="Bar open",marker=dict(color=C_GREEN,size=5,symbol="triangle-up")))
fig_px.add_trace(go.Scatter(x=bars_df["close_time"],y=bars_df["close"],
    mode="markers",name="Bar close",marker=dict(color=C_RED,size=5,symbol="triangle-down")))
fig_px.update_layout(template=PT,title="Price with FIB Bar Boundaries",
    xaxis_title="Time (s)",yaxis_title="Price",height=320,
    margin=dict(l=50,r=20,t=40,b=40),legend=dict(orientation="h",y=-0.18))
st.plotly_chart(fig_px, use_container_width=True)

ca, cb = st.columns(2)
CRMAP = {"threshold":C_GREEN,"timeout":C_ORANGE,"flush":C_GREY,"inactivity":C_PURPLE,"max_events":C_RED}

with ca:
    fig_dur = px.histogram(bars_df,x="duration_seconds",nbins=40,color="close_reason",
        color_discrete_map=CRMAP,title="Bar Duration Distribution",template=PT,
        labels={"duration_seconds":"Duration (s)","close_reason":"Close reason"})
    fig_dur.update_layout(height=290,margin=dict(l=40,r=10,t=40,b=40))
    st.plotly_chart(fig_dur, use_container_width=True)

with cb:
    fig_ne = px.histogram(bars_df,x="n_events",nbins=40,
        title="Events per Bar",template=PT,labels={"n_events":"Events/bar"})
    fig_ne.update_traces(marker_color=C_BLUE)
    fig_ne.update_layout(height=290,margin=dict(l=40,r=10,t=40,b=40))
    st.plotly_chart(fig_ne, use_container_width=True)

# Info scalar vs threshold
fig_info = make_subplots(specs=[[{"secondary_y":True}]])
fig_info.add_trace(go.Scatter(x=aug_df["bar_index"],y=aug_df["information_scalar"],
    name="Φ(J) scalar",line=dict(color=C_BLUE,width=1.5)),secondary_y=False)
fig_info.add_trace(go.Scatter(x=aug_df["bar_index"],y=aug_df["threshold_at_close"],
    name="I* threshold",line=dict(color=C_ORANGE,width=1.5,dash="dash")),secondary_y=False)
fig_info.add_trace(go.Bar(x=aug_df["bar_index"],y=aug_df["threshold_utilization"],
    name="Utilization",marker_color=C_PURPLE,opacity=0.35),secondary_y=True)
fig_info.update_layout(template=PT,title="Information Scalar vs Threshold",
    xaxis_title="Bar index",height=310,margin=dict(l=50,r=50,t=40,b=50),
    legend=dict(orientation="h",y=-0.22))
fig_info.update_yaxes(title_text="Scalar / Threshold",secondary_y=False)
fig_info.update_yaxes(title_text="Utilization",secondary_y=True)
st.plotly_chart(fig_info, use_container_width=True)

cc, cd = st.columns(2)
with cc:
    fig_rate = go.Figure(go.Scatter(x=aug_df["bar_index"],y=aug_df["information_rate"],
        mode="lines+markers",line=dict(color=C_GREEN,width=1.5),marker=dict(size=4)))
    fig_rate.update_layout(template=PT,title="Information Rate (Φ/s)",
        xaxis_title="Bar index",yaxis_title="Info/s",height=280,margin=dict(l=50,r=10,t=40,b=40))
    st.plotly_chart(fig_rate, use_container_width=True)

with cd:
    rc = bars_df["close_reason"].value_counts()
    fig_pie = px.pie(values=rc.values,names=rc.index,color=rc.index,
        color_discrete_map=CRMAP,title="Close Reason Distribution",template=PT,hole=0.4)
    fig_pie.update_layout(height=280,margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

if bl_df is not None and not bl_df.empty:
    st.markdown("**FIB vs Baseline**")
    ce, cf = st.columns(2)
    with ce:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Histogram(x=bars_df["n_events"],name="FIB",
            opacity=0.65,marker_color=C_BLUE,nbinsx=30))
        fig_cmp.add_trace(go.Histogram(x=bl_df["n_events"],
            name=f"Baseline ({bl_type})",opacity=0.65,marker_color=C_ORANGE,nbinsx=30))
        fig_cmp.update_layout(barmode="overlay",template=PT,
            title="Events/Bar: FIB vs Baseline",xaxis_title="Events",
            height=280,margin=dict(l=40,r=10,t=40,b=40),legend=dict(orientation="h",y=-0.22))
        st.plotly_chart(fig_cmp, use_container_width=True)
    with cf:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=bars_df["duration_seconds"],name="FIB",
            marker_color=C_BLUE,boxmean=True))
        fig_box.add_trace(go.Box(y=bl_df["duration_seconds"],
            name=f"Baseline ({bl_type})",marker_color=C_ORANGE,boxmean=True))
        fig_box.update_layout(template=PT,title="Duration: FIB vs Baseline",
            yaxis_title="Duration (s)",height=280,margin=dict(l=40,r=10,t=40,b=40))
        st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Downloads ─────────────────────────────────────────────────────────────────
st.subheader("5 · Downloads")
d1,d2,d3,d4 = st.columns(4)
with d1:
    st.download_button("⬇ FIB Bars CSV", data=_csv(bars_df),
        file_name="fib_bars.csv", mime="text/csv")
with d2:
    st.download_button("⬇ Augmented CSV", data=_csv(aug_df),
        file_name="fib_augmented.csv", mime="text/csv")
with d3:
    exp = {k:v for k,v in sm.items() if k!="reasons"}
    exp.update(sm.get("reasons",{}))
    exp["config"] = {"model":model,"info_mode":info_mode,"scalarizer":scalarizer,
                     "eta":eta,"delta0_seconds":delta0,"timeout_seconds":float(timeout_s)}
    st.download_button("⬇ Summary JSON",
        data=json.dumps(exp,indent=2).encode("utf-8"),
        file_name="fib_summary.json", mime="application/json")
with d4:
    if bl_df is not None and not bl_df.empty:
        st.download_button(f"⬇ Baseline CSV", data=_csv(bl_df),
            file_name=f"baseline_{bl_type}.csv", mime="text/csv")
    else:
        st.button("⬇ Baseline CSV", disabled=True)

st.markdown(
    "<div style='text-align:center;color:#555;font-size:.75rem;margin-top:1.5rem'>"
    "Fisher Information Bars v1.1.0 · "
    "A bar closes when ∑ Ψᵢ(θ̂) ≥ I* — not when a clock ticks."
    "</div>", unsafe_allow_html=True)