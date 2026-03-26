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
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP  v3.0  —  professional light theme, review-driven rebuild
# ═══════════════════════════════════════════════════════════════════════════════
import io, json, traceback
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Fisher Information Bars",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens (light, professional) ─────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}
.block-container {
    padding: 1.6rem 2rem 2rem;
    max-width: 1360px;
}
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F8F9FB;
    border-right: 1px solid #E4E7ED;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.76rem;
    color: #6B7280;
    line-height: 1.5;
}
/* Sidebar section labels */
.sb-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    border-bottom: 1px solid #E4E7ED;
    padding-bottom: 4px;
    margin: 14px 0 8px;
}
/* ── Page header ── */
.page-title {
    font-size: 1.45rem;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}
.page-sub {
    font-size: 0.82rem;
    color: #6B7280;
    font-family: 'JetBrains Mono', monospace;
}
/* ── Section headings ── */
.section-heading {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 1.6rem 0 0.6rem;
    border-bottom: 1px solid #F3F4F6;
    padding-bottom: 5px;
}
/* ── KPI cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 10px;
}
.kpi {
    background: #FFFFFF;
    border: 1px solid #E4E7ED;
    border-radius: 8px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 5px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    color: #111827;
    line-height: 1;
    margin-bottom: 3px;
}
.kpi-sub {
    font-size: 0.7rem;
    color: #9CA3AF;
}
.kpi-accent .kpi-value { color: #1D4ED8; }
.kpi-warn   .kpi-value { color: #B45309; }
.kpi-good   .kpi-value { color: #065F46; }
/* ── Callouts ── */
.callout {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    border-radius: 6px;
    padding: 9px 12px;
    font-size: 0.78rem;
    margin: 6px 0;
    line-height: 1.5;
}
.callout-warn  { background: #FFFBEB; border: 1px solid #FCD34D; color: #92400E; }
.callout-info  { background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A8A; }
.callout-good  { background: #ECFDF5; border: 1px solid #6EE7B7; color: #065F46; }
/* ── Preset buttons ── */
.preset-active {
    border-bottom: 2px solid #1D4ED8 !important;
    color: #1D4ED8 !important;
    font-weight: 600 !important;
}
/* ── Legend pills ── */
.legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 10px; }
.legend-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.72rem; color: #374151;
    background: #F9FAFB; border: 1px solid #E4E7ED;
    border-radius: 20px; padding: 3px 9px;
}
.legend-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Consistent colour palette (concept-mapped) ────────────────────────────────
C = {
    "threshold":  "#1D4ED8",   # blue  — information threshold close
    "timeout":    "#D97706",   # amber — timeout/safety-valve close
    "flush":      "#9CA3AF",   # grey  — end-of-data flush
    "inactivity": "#7C3AED",   # violet — inactivity close
    "max_events": "#DC2626",   # red   — max-events close
    "fib":        "#1D4ED8",   # blue  — FIB series
    "baseline":   "#D97706",   # amber — baseline series
    "price":      "#374151",   # charcoal — price line
    "scalar":     "#1D4ED8",   # blue  — info scalar
    "threshold_line": "#D97706", # amber — threshold line
    "rate":       "#059669",   # green — information rate
}
CRMAP = {k: C[k] for k in ["threshold","timeout","flush","inactivity","max_events"]}
PT = "plotly_white"


# ── Utility functions ─────────────────────────────────────────────────────────

def _sb_label(t):
    st.sidebar.markdown(f'<div class="sb-label">{t}</div>', unsafe_allow_html=True)

def _sh(t):
    st.markdown(f'<div class="section-heading">{t}</div>', unsafe_allow_html=True)

def _callout(msg, kind="warn"):
    icon = {"warn": "⚠", "info": "ℹ", "good": "✓"}.get(kind, "ℹ")
    st.markdown(
        f'<div class="callout callout-{kind}"><span>{icon}</span><span>{msg}</span></div>',
        unsafe_allow_html=True,
    )

def _csv(df):
    return df.to_csv(index=False).encode("utf-8")

def _coerce_ts(df, col):
    df = df.copy()
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        df[col] = s.astype(float)
        return df
    try:
        df[col] = pd.to_datetime(s, utc=True).astype("int64") / 1e9
        return df
    except Exception:
        pass
    try:
        df[col] = s.astype(float)
    except Exception:
        pass
    return df

def _summary(df):
    if df.empty:
        return {}
    n = len(df)
    ne = df["n_events"]
    dur = df["duration_seconds"]
    info = df["information_scalar"]
    thr = (df["close_reason"] == "threshold").sum()
    cv = ne.std() / ne.mean() if ne.mean() > 0 else float("nan")
    return {
        "n_bars": n, "avg_events": float(ne.mean()), "cv_events": float(cv),
        "avg_dur": float(dur.mean()), "med_dur": float(dur.median()),
        "pct_thr": float(thr / n * 100), "avg_info": float(info.mean()),
        "med_info": float(info.median()),
        "reasons": df["close_reason"].value_counts().to_dict(),
    }

def _kpi(label, value, sub="", style=""):
    return (
        f'<div class="kpi {style}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )

def _legend(*items):
    pills = "".join(
        f'<span class="legend-pill">'
        f'<span class="legend-dot" style="background:{c}"></span>{label}</span>'
        for label, c in items
    )
    st.markdown(f'<div class="legend-row">{pills}</div>', unsafe_allow_html=True)

def _synthetic(n=2000, seed=42, vol_regime="mixed", drift=0.0):
    rng = np.random.default_rng(seed)
    price = 100.0
    prices, ts, sizes = [], [], []
    t = 0.0
    vmap = {
        "calm":     ([0.02], [1.0]),
        "mixed":    ([0.03, 0.15, 0.55], [0.70, 0.22, 0.08]),
        "volatile": ([0.08, 0.35, 0.90], [0.50, 0.35, 0.15]),
    }
    vols, probs = vmap.get(vol_regime, vmap["mixed"])
    for _ in range(n):
        vol = rng.choice(vols, p=probs)
        price = max(price * np.exp(drift / n + rng.normal(0, vol)), 1.0)
        t += rng.exponential(0.5)
        prices.append(round(price, 4))
        ts.append(round(t, 4))
        sizes.append(int(rng.integers(1, 20)))
    df = pd.DataFrame({"timestamp": ts, "price": prices, "size": sizes})
    h = 0.015 + 0.008 * rng.random(n)
    df["bid"] = (df["price"] - h).round(4)
    df["ask"] = (df["price"] + h).round(4)
    return df


# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "Fast": {
        "eta": 0.3, "delta0": 15.0, "ewma_a": 0.15, "min_wu": 3,
        "timeout_s": 60.0, "max_ev": 500, "model": "gaussian",
        "_desc": "Short, frequent bars. High alpha for fast threshold adaptation. "
                 "Good for exploring dense tick data.",
    },
    "Balanced": {
        "eta": 1.0, "delta0": 60.0, "ewma_a": 0.05, "min_wu": 10,
        "timeout_s": 300.0, "max_ev": 5000, "model": "garch",
        "_desc": "Sensible defaults for most equity/FX datasets. "
                 "GARCH adapts to volatility clustering.",
    },
    "Robust": {
        "eta": 2.5, "delta0": 120.0, "ewma_a": 0.02, "min_wu": 20,
        "timeout_s": 600.0, "max_ev": 10000, "model": "garch",
        "_desc": "Larger bars with stable thresholds. Suited for illiquid or "
                 "sparse data where patience is required.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◈ Fisher Information Bars")
    st.markdown("Adaptive bars close when accumulated statistical information reaches a target — not when a clock ticks.")

    # ── Presets ───────────────────────────────────────────────────────────────
    _sb_label("PRESETS")
    preset_cols = st.columns(3)
    for i, pname in enumerate(PRESETS):
        if preset_cols[i].button(pname, key=f"pre_{i}", use_container_width=True,
                                 help=PRESETS[pname]["_desc"]):
            st.session_state["preset"] = pname
    active = st.session_state.get("preset", "Balanced")
    p = PRESETS[active]
    st.caption(f"**{active}:** {p['_desc']}")

    # ── Data source ───────────────────────────────────────────────────────────
    _sb_label("DATA SOURCE")
    data_source = st.radio("Source", ["Upload CSV", "Synthetic data"],
                           label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload tick data (CSV)",
            type=["csv"],
            help="Required columns: timestamp (Unix seconds or ISO datetime), price.\n"
                 "Optional: size, bid, ask.",
        )
    else:
        n_synth    = st.slider("Number of events", 500, 10_000, 2_000, 500)
        vol_regime = st.selectbox(
            "Volatility regime", ["calm", "mixed", "volatile"], index=1,
            help="calm: low, stable volatility.\n"
                 "mixed: regime-switching (most realistic for equity/FX).\n"
                 "volatile: frequent high-vol stress episodes.",
        )
        drift_pct = st.slider(
            "Annual drift (%)", -50, 50, 0, 5,
            help="Log-drift applied to the simulated price path per year.",
        ) / 100
        synth_seed = st.number_input("Random seed", value=42, step=1,
                                     help="Fix to reproduce the same synthetic path.")

    # ── Model ─────────────────────────────────────────────────────────────────
    _sb_label("MODEL")
    model_opts = ["gaussian", "garch", "hawkes"]
    model = st.selectbox(
        "Statistical model",
        model_opts,
        index=model_opts.index(p.get("model", "garch")),
        help=(
            "gaussian — Local N(μ,σ²). Simple, fast. Best for near-normal returns.\n\n"
            "garch — GARCH(1,1) quasi-likelihood. Adapts σ² to volatility clustering. "
            "Best for equity and FX data.\n\n"
            "hawkes — Self-exciting point process. Information comes from arrival "
            "intensity λ(t), not price. Best for order-flow or event-driven data."
        ),
    )
    info_mode = st.selectbox(
        "Information mode", ["observed", "expected"],
        help=(
            "observed — Outer product of score gradients (OPG). Reacts to the "
            "realised market path. Recommended for most use cases.\n\n"
            "expected — Analytic Fisher information. Smoother but less responsive "
            "to sudden regime changes."
        ),
    )
    scalarizer = st.selectbox(
        "Scalarizer", ["logdet", "trace", "frobenius"],
        help=(
            "How to reduce the n×n information matrix J to a single number:\n\n"
            "logdet — log det(J + εI). Reparameterisation-invariant: the same bar "
            "forms regardless of how parameters are scaled. Recommended.\n\n"
            "trace — tr(J + εI). Fast and simple; scale-sensitive.\n\n"
            "frobenius — ‖J‖_F. Captures off-diagonal structure; not invariant."
        ),
    )

    # ── Model-specific parameters (only show relevant ones) ───────────────────
    with st.expander(f"{model.upper()} model parameters", expanded=False):
        if model == "gaussian":
            var_floor = st.select_slider(
                "Variance floor",
                options=[1e-14, 1e-12, 1e-10, 1e-8, 1e-6],
                value=1e-12,
                format_func=lambda x: f"{x:.0e}",
                help="Minimum σ². Prevents division-by-zero when price is perfectly constant.",
            )
            garch_p, hawkes_fl = 0.9999, 1e-8
        elif model == "garch":
            garch_p = st.select_slider(
                "Max persistence α+β",
                options=[0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 0.9999],
                value=0.9999,
                format_func=lambda x: f"{x:.4f}",
                help="Cap on α+β for stationarity. Lower values force faster "
                     "mean-reversion of the conditional variance. Must be < 1.",
            )
            var_floor, hawkes_fl = 1e-12, 1e-8
        else:
            hawkes_fl = st.select_slider(
                "Intensity floor",
                options=[1e-10, 1e-8, 1e-6, 1e-4],
                value=1e-8,
                format_func=lambda x: f"{x:.0e}",
                help="Minimum λ(t). Prevents log(0) during periods with no arrivals.",
            )
            var_floor, garch_p = 1e-12, 0.9999

        eps_ridge = st.select_slider(
            "Ridge ε",
            options=[1e-8, 1e-6, 1e-4, 1e-2],
            value=1e-6,
            format_func=lambda x: f"{x:.0e}",
            help="Added to J before scalarization: Φ(J + εI). Stabilises computation "
                 "when the information matrix is nearly singular.",
        )

    # ── Adaptive threshold ────────────────────────────────────────────────────
    _sb_label("ADAPTIVE THRESHOLD")
    c1, c2 = st.columns([3, 1])
    eta = c1.slider("η multiplier", 0.1, 10.0, float(p["eta"]), 0.1,
                    help="Scales the information quantum I*. Higher → fewer, larger bars. "
                         "Lower → more frequent bars. Start at 1.0 and halve if too few bars form.")
    eta = float(c2.number_input("η", value=eta, min_value=0.01, max_value=20.0,
                                format="%.2f", label_visibility="collapsed", key="eta_num"))

    c3, c4 = st.columns([3, 1])
    delta0 = c3.slider("δ₀ reference duration (s)", 1.0, 600.0, float(p["delta0"]), 1.0,
                       help="Target bar duration in seconds. At equilibrium, average bar "
                            "duration ≈ δ₀. Set to your desired typical bar length.")
    delta0 = float(c4.number_input("δ₀", value=delta0, min_value=0.1, max_value=3600.0,
                                   format="%.1f", label_visibility="collapsed", key="d0_num"))

    c5, c6 = st.columns([3, 1])
    ewma_a = c5.slider("EWMA α", 0.01, 0.50, float(p["ewma_a"]), 0.01,
                       help="Smoothing for the long-run information rate estimate. "
                            "Lower = more stable threshold. Higher = faster adaptation to regime changes.")
    ewma_a = float(c6.number_input("α", value=ewma_a, min_value=0.001, max_value=1.0,
                                   format="%.3f", label_visibility="collapsed", key="a_num"))

    min_wu = st.number_input(
        "Warmup bars", min_value=1, max_value=100, value=int(p["min_wu"]),
        help="Number of completed bars before the EWMA threshold is trusted. "
             "During warmup, I* seeds from the first observed scalar. "
             "Use 3–5 for short datasets, 20+ for stable long-run estimation.",
    )

    # ── Safety valves ─────────────────────────────────────────────────────────
    with st.expander("Safety valves & timeouts", expanded=False):
        timeout_s = st.number_input(
            "Bar timeout (s)", min_value=1.0, value=float(p["timeout_s"]), step=10.0,
            help="Maximum wall-clock seconds any bar may stay open. "
                 "Prevents bar starvation in dead markets.",
        )
        use_inact = st.checkbox(
            "Inactivity timeout",
            help="Close a bar when no new tick has arrived for N seconds. "
                 "Useful for illiquid assets with extended quiet periods.",
        )
        inact_s: Optional[float] = None
        if use_inact:
            inact_s = st.number_input(
                "Inactivity gap (s)", min_value=1.0, value=60.0, step=5.0,
                help="Seconds of silence that trigger an inactivity bar close.",
            )
        max_ev = st.number_input(
            "Max events per bar", min_value=10, value=int(p["max_ev"]), step=100,
            help="Hard cap on ticks per bar. A safety valve against runaway "
                 "accumulation in low-volatility environments.",
        )

    # ── Baseline comparison ───────────────────────────────────────────────────
    with st.expander("Baseline comparison", expanded=False):
        show_bl = st.checkbox("Compute baseline bars", value=True,
                              help="Build a conventional bar scheme to compare against FIBs.")
        bl_type = st.selectbox(
            "Baseline type", ["time", "tick", "volume", "dollar"],
            help="time: fixed wall-clock duration.\ntick: fixed event count.\n"
                 "volume: fixed cumulative size.\ndollar: fixed cumulative price×size.",
        )
        if bl_type == "time":     bl_p = st.number_input("Seconds / bar",  min_value=1.0,  value=60.0)
        elif bl_type == "tick":   bl_p = st.number_input("Ticks / bar",    min_value=2,    value=50, step=5)
        elif bl_type == "volume": bl_p = st.number_input("Volume / bar",   min_value=1.0,  value=1000.0)
        else:                     bl_p = st.number_input("Dollar / bar",   min_value=1.0,  value=100_000.0)

    st.divider()
    run_btn = st.button("◈  Build FIB Bars", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="page-title">Fisher Information Bars</div>'
    '<div class="page-sub">Bar closes when Σ Ψᵢ(θ̂) ≥ I* &nbsp;·&nbsp; '
    'information-geometric financial sampling</div>',
    unsafe_allow_html=True,
)

# ── Load / generate data ───────────────────────────────────────────────────────
raw_df = None
if data_source == "Upload CSV":
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
else:
    raw_df = _synthetic(n=n_synth, seed=int(synth_seed),
                        vol_regime=vol_regime, drift=drift_pct)

# ── Landing state ─────────────────────────────────────────────────────────────
if raw_df is None:
    st.markdown("<br>", unsafe_allow_html=True)
    _callout(
        "Upload a CSV file using the sidebar, then click <strong>◈ Build FIB Bars</strong>.",
        "info",
    )
    with st.expander("Expected CSV format", expanded=True):
        st.markdown("""
| Column | Required | Type | Notes |
|--------|----------|------|-------|
| `timestamp` | ✅ | float / datetime | Unix seconds or ISO 8601 string |
| `price` | ✅ | float | Last trade price or mid-quote |
| `size` | optional | float | Trade size / contracts |
| `bid` | optional | float | Best bid — enables spread features |
| `ask` | optional | float | Best ask — enables spread features |
""")
    st.stop()

# ── Section 1: Data preview ────────────────────────────────────────────────────
_sh("01 · DATA PREVIEW")

cols = list(raw_df.columns)

# Auto-detect columns
def _find(cols, patterns, default=0):
    for i, c in enumerate(cols):
        if any(p in c.lower() for p in patterns):
            return i
    return default

if data_source == "Upload CSV":
    with st.expander("Column mapping", expanded=True):
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        ts_col  = mc1.selectbox("timestamp", cols, index=_find(cols, ["time","ts","date"]))
        px_col  = mc2.selectbox("price",     cols, index=_find(cols, ["price","mid","close","last"]))
        sz_col  = mc3.selectbox("size (opt)", ["(none)"] + cols,
                                index=_find(["(none)"]+cols, ["size","qty","vol","amount"])+0)
        bid_col = mc4.selectbox("bid (opt)",  ["(none)"] + cols,
                                index=_find(["(none)"]+cols, ["bid"])+0)
        ask_col = mc5.selectbox("ask (opt)",  ["(none)"] + cols,
                                index=_find(["(none)"]+cols, ["ask","offer"])+0)
else:
    # Synthetic — columns are fixed
    ts_col = "timestamp"; px_col = "price"
    sz_col = "size"; bid_col = "bid"; ask_col = "ask"

mapped: dict = {"timestamp": raw_df[ts_col], "price": raw_df[px_col]}
if sz_col  not in ("(none)", None) and sz_col  in raw_df.columns: mapped["size"] = raw_df[sz_col]
if bid_col not in ("(none)", None) and bid_col in raw_df.columns: mapped["bid"]  = raw_df[bid_col]
if ask_col not in ("(none)", None) and ask_col in raw_df.columns: mapped["ask"]  = raw_df[ask_col]
mapped_df = _coerce_ts(pd.DataFrame(mapped), "timestamp")

# Synthetic summary strip
if data_source == "Synthetic data":
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Events",       f"{len(mapped_df):,}")
    sc2.metric("Price min",    f"{mapped_df['price'].min():.2f}")
    sc3.metric("Price max",    f"{mapped_df['price'].max():.2f}")
    sc4.metric("Duration",     f"{mapped_df['timestamp'].max():.1f}s")
    sc5.metric("Regime",       vol_regime)
    st.caption(
        f"Regime-switching GBM · volatility={vol_regime} · "
        f"drift={drift_pct*100:+.0f}%/yr · seed={synth_seed}. "
        "Timestamps are elapsed seconds from session start."
    )

st.dataframe(mapped_df.head(200), use_container_width=True, height=175)
st.caption(f"{len(mapped_df):,} rows · mapped columns: {list(mapped_df.columns)}")

if not run_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    _callout(
        "Configure parameters in the sidebar, then click "
        "<strong>◈ Build FIB Bars</strong> to run.",
        "info",
    )
    st.stop()

# ── Build FIB bars ─────────────────────────────────────────────────────────────
try:
    cfg = FIBConfig(
        model=model, info_mode=info_mode, scalarizer=scalarizer,
        eta=float(eta), delta0_seconds=float(delta0), ewma_alpha=float(ewma_a),
        eps_ridge=float(eps_ridge), timeout_seconds=float(timeout_s),
        max_events_per_bar=int(max_ev), inactivity_timeout_seconds=inact_s,
        min_warmup_events=int(min_wu), var_floor=float(var_floor),
        garch_persistence_max=float(garch_p), hawkes_intensity_floor=float(hawkes_fl),
    )
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

with st.spinner("Building FIB bars…"):
    try:
        bars_df = build_fib_bars(mapped_df, config=cfg)
    except Exception:
        st.error("FIB build failed — see traceback below.")
        st.code(traceback.format_exc())
        st.stop()

if bars_df.empty:
    _callout(
        "No bars produced. Try reducing η or δ₀, or increasing the bar timeout.",
        "warn",
    )
    st.stop()

aug_df = augment_with_fib_features(bars_df)
sm = _summary(bars_df)

# Baseline
bl_df = None
if show_bl:
    with st.spinner("Building baseline bars…"):
        try:
            kw = {}
            if bl_type == "time":     kw = {"seconds_per_bar": float(bl_p)}
            elif bl_type == "tick":   kw = {"ticks_per_bar": int(bl_p)}
            elif bl_type == "volume": kw = {"volume_per_bar": float(bl_p)}
            else:                     kw = {"dollar_per_bar": float(bl_p)}
            bl_df = build_baseline_bars(mapped_df, bar_type=bl_type, **kw)
        except Exception:
            st.warning("Baseline build failed — comparison unavailable.")
bsm = _summary(bl_df) if bl_df is not None and not bl_df.empty else {}

# ── Automatic diagnostics ──────────────────────────────────────────────────────
timeout_pct = 100.0 - sm["pct_thr"]
if timeout_pct > 60:
    _callout(
        f"<strong>{timeout_pct:.0f}%</strong> of bars closed via timeout/flush rather than "
        "the information threshold. The threshold may be set too high. "
        "Try reducing <strong>η</strong> or <strong>δ₀</strong>.",
        "warn",
    )
if sm.get("cv_events", 0) > 1.5:
    _callout(
        f"CV of events/bar = <strong>{sm['cv_events']:.2f}</strong> (high). "
        "Bars are very uneven in size. Consider lowering EWMA α for a more stable threshold, "
        "or switching to GARCH if the data has volatility clustering.",
        "warn",
    )
if "size" not in mapped_df.columns:
    _callout("size column not mapped — volume and dollar-value features will be zero.", "warn")
if "bid" not in mapped_df.columns or "ask" not in mapped_df.columns:
    _callout("bid/ask columns not mapped — spread features will be unavailable.", "warn")
if sm["n_bars"] < 5:
    _callout(
        f"Only <strong>{sm['n_bars']}</strong> bars produced. The first bar seeds the "
        "threshold; very few bars may indicate η is too high or data is too short. "
        "Try the <strong>Fast</strong> preset or reduce η.",
        "warn",
    )
if timeout_pct == 0 and sm["n_bars"] > 3:
    _callout(
        "All bars closed on the information threshold. The algorithm is working well.",
        "good",
    )

st.divider()

# ── Section 2: Summary metrics ─────────────────────────────────────────────────
_sh("02 · SUMMARY METRICS")

thr_style = "kpi-good" if sm["pct_thr"] >= 50 else "kpi-warn"
r1 = "".join([
    _kpi("FIB bars",          f"{sm['n_bars']:,}",
         f"vs {bsm.get('n_bars','—')} {bl_type} baseline" if bsm else "", "kpi-accent"),
    _kpi("Avg events / bar",  f"{sm['avg_events']:.0f}",
         f"CV = {sm['cv_events']:.2f}"),
    _kpi("Avg duration",      f"{sm['avg_dur']:.1f}s",
         f"median {sm['med_dur']:.1f}s"),
    _kpi("Threshold closes",  f"{sm['pct_thr']:.0f}%",
         "bars that hit I*", thr_style),
])
r2 = "".join([
    _kpi("Avg info scalar",   f"{sm['avg_info']:.3g}",
         f"median {sm['med_info']:.3g}"),
    _kpi("Model",             model.upper(), info_mode),
    _kpi("Scalarizer",        scalarizer, f"ε = {eps_ridge:.0e}"),
    _kpi("η / δ₀",            f"{eta} / {delta0}s", f"EWMA α = {ewma_a}"),
])
st.markdown(
    f'<div class="kpi-row">{r1}</div><div class="kpi-row">{r2}</div>',
    unsafe_allow_html=True,
)

if bsm:
    bc1, bc2, bc3, bc4 = st.columns(4)
    cv_delta = sm["cv_events"] - bsm["cv_events"]
    bc1.metric("Baseline bars",          f"{bsm['n_bars']:,}",
               f"{bsm['n_bars'] - sm['n_bars']:+,} vs FIB")
    bc2.metric("Avg events / bar (BL)",  f"{bsm['avg_events']:.0f}")
    bc3.metric("CV events / bar (FIB)",  f"{sm['cv_events']:.2f}",
               f"{cv_delta:+.2f} vs baseline", delta_color="inverse")
    bc4.metric("Avg duration (BL)",      f"{bsm['avg_dur']:.1f}s")

st.divider()

# ── Section 3: Charts ──────────────────────────────────────────────────────────
_sh("03 · CHARTS")

# ── 3a: Price series ──────────────────────────────────────────────────────────
_legend(
    ("Threshold close", C["threshold"]),
    ("Timeout / flush", C["timeout"]),
    ("Bar open", C["threshold"]),
    ("Bar close", C["timeout"]),
)
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=mapped_df["timestamp"], y=mapped_df["price"],
    mode="lines", name="Price",
    line=dict(color=C["price"], width=1),
))

# Shade timeout/flush regions
for _, row in bars_df[bars_df["close_reason"].isin(
        ["timeout", "inactivity", "max_events", "flush"])].iterrows():
    fig_price.add_vrect(
        x0=row["open_time"], x1=row["close_time"],
        fillcolor=C["timeout"], opacity=0.07, line_width=0,
    )

# Threshold-close vertical marks
for _, row in bars_df[bars_df["close_reason"] == "threshold"].iterrows():
    fig_price.add_vline(
        x=row["close_time"],
        line_color=C["threshold"], line_width=0.9, opacity=0.45,
    )

hover_idx = aug_df["bar_index"].values if "bar_index" in aug_df.columns \
    else np.arange(len(bars_df))

fig_price.add_trace(go.Scatter(
    x=bars_df["open_time"], y=bars_df["open"],
    mode="markers", name="Bar open",
    marker=dict(color=C["threshold"], size=6, symbol="triangle-up",
                line=dict(color="white", width=1)),
    hovertemplate="Open — bar %{customdata}<br>t = %{x:.2f}s<br>price = %{y:.4f}<extra></extra>",
    customdata=hover_idx,
))
fig_price.add_trace(go.Scatter(
    x=bars_df["close_time"], y=bars_df["close"],
    mode="markers", name="Bar close",
    marker=dict(
        color=[C.get(r, C["flush"]) for r in bars_df["close_reason"]],
        size=6, symbol="triangle-down",
        line=dict(color="white", width=1),
    ),
    customdata=np.column_stack([
        hover_idx,
        bars_df["close_reason"].values,
        bars_df["information_scalar"].values,
    ]),
    hovertemplate=(
        "Close — bar %{customdata[0]}<br>"
        "t = %{x:.2f}s<br>price = %{y:.4f}<br>"
        "reason = %{customdata[1]}<br>"
        "Φ(J) = %{customdata[2]:.4f}<extra></extra>"
    ),
))
fig_price.update_layout(
    template=PT, height=320,
    title="Price with FIB bar boundaries",
    xaxis_title="Time (s)", yaxis_title="Price",
    margin=dict(l=50, r=20, t=42, b=40),
    legend=dict(orientation="h", y=-0.2, font_size=11),
    plot_bgcolor="white",
)
st.plotly_chart(fig_price, use_container_width=True)

# ── 3b: Duration + events histograms ─────────────────────────────────────────
col_a, col_b = st.columns(2)
nbins = min(40, max(10, sm["n_bars"]))

with col_a:
    fig_dur = px.histogram(
        bars_df, x="duration_seconds", nbins=nbins,
        color="close_reason", color_discrete_map=CRMAP,
        template=PT,
        labels={"duration_seconds": "Duration (s)", "close_reason": "Close reason"},
        title="Bar duration",
    )
    fig_dur.update_layout(
        height=270, bargap=0.05, legend_title_text="",
        margin=dict(l=45, r=10, t=42, b=40),
        plot_bgcolor="white",
        xaxis_title="Duration (s)", yaxis_title="Count",
    )
    st.plotly_chart(fig_dur, use_container_width=True)

with col_b:
    fig_ne = px.histogram(
        bars_df, x="n_events", nbins=nbins,
        color="close_reason", color_discrete_map=CRMAP,
        template=PT,
        labels={"n_events": "Events per bar", "close_reason": "Close reason"},
        title="Events per bar",
    )
    fig_ne.update_layout(
        height=270, bargap=0.05, legend_title_text="",
        margin=dict(l=45, r=10, t=42, b=40),
        plot_bgcolor="white",
        xaxis_title="Events per bar", yaxis_title="Count",
    )
    st.plotly_chart(fig_ne, use_container_width=True)

# ── 3c: Information scalar vs threshold ──────────────────────────────────────
fig_info = make_subplots(specs=[[{"secondary_y": True}]])
fig_info.add_trace(go.Scatter(
    x=aug_df["bar_index"], y=aug_df["information_scalar"],
    name="Φ(J) — accumulated information",
    line=dict(color=C["scalar"], width=2),
    hovertemplate="Bar %{x}<br>Φ(J) = %{y:.4f}<extra></extra>",
), secondary_y=False)
fig_info.add_trace(go.Scatter(
    x=aug_df["bar_index"], y=aug_df["threshold_at_close"],
    name="I* — adaptive threshold",
    line=dict(color=C["threshold_line"], width=1.5, dash="dash"),
    hovertemplate="Bar %{x}<br>I* = %{y:.4f}<extra></extra>",
), secondary_y=False)
util_clipped = aug_df["threshold_utilization"].clip(-2, 2)
fig_info.add_trace(go.Bar(
    x=aug_df["bar_index"], y=util_clipped,
    name="Utilisation Φ/I*",
    marker=dict(
        color=util_clipped,
        colorscale=[[0, "#FCA5A5"], [0.4, "#E5E7EB"], [1, "#93C5FD"]],
        cmin=0, cmax=1.2,
    ),
    opacity=0.5,
    hovertemplate="Bar %{x}<br>Utilisation = %{y:.2f}<extra></extra>",
), secondary_y=True)
fig_info.update_layout(
    template=PT, height=300,
    title="Information scalar Φ(J) vs adaptive threshold I*",
    xaxis_title="Bar index",
    margin=dict(l=50, r=65, t=42, b=50),
    legend=dict(orientation="h", y=-0.24, font_size=11),
    plot_bgcolor="white",
)
fig_info.update_yaxes(title_text="Φ(J) / I*", secondary_y=False)
fig_info.update_yaxes(title_text="Utilisation ratio", secondary_y=True, showgrid=False)
st.plotly_chart(fig_info, use_container_width=True)

# ── 3d: Information rate + close-reason donut ─────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    fig_rate = go.Figure(go.Scatter(
        x=aug_df["bar_index"], y=aug_df["information_rate"],
        mode="lines+markers",
        line=dict(color=C["rate"], width=1.5),
        marker=dict(size=5, color=C["rate"]),
        hovertemplate="Bar %{x}<br>Rate = %{y:.4f} Φ/s<extra></extra>",
    ))
    fig_rate.update_layout(
        template=PT, height=265,
        title="Information rate (Φ per second)",
        xaxis_title="Bar index", yaxis_title="Φ / s",
        margin=dict(l=50, r=10, t=42, b=40),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_rate, use_container_width=True)

with col_d:
    rc = bars_df["close_reason"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=rc.index, values=rc.values,
        marker=dict(
            colors=[CRMAP.get(r, "#9CA3AF") for r in rc.index],
            line=dict(color="white", width=2),
        ),
        hole=0.52,
        textinfo="label+percent",
        textfont=dict(size=11, color="#374151"),
        hovertemplate="%{label}: %{value} bars (%{percent})<extra></extra>",
    ))
    fig_pie.update_layout(
        template=PT, height=265,
        title="Bar close reasons",
        margin=dict(l=10, r=10, t=42, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── 3e: FIB vs Baseline ───────────────────────────────────────────────────────
if bl_df is not None and not bl_df.empty:
    _sh("04 · FIB vs BASELINE")
    _legend(("FIB", C["fib"]), (f"Baseline ({bl_type})", C["baseline"]))
    col_e, col_f = st.columns(2)

    with col_e:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Histogram(
            x=bars_df["n_events"], name="FIB",
            opacity=0.75, marker_color=C["fib"], nbinsx=30,
        ))
        fig_cmp.add_trace(go.Histogram(
            x=bl_df["n_events"], name=f"Baseline ({bl_type})",
            opacity=0.55, marker_color=C["baseline"], nbinsx=30,
        ))
        fig_cmp.update_layout(
            barmode="overlay", template=PT, height=265,
            title="Events per bar: FIB vs baseline",
            xaxis_title="Events per bar", yaxis_title="Count",
            margin=dict(l=45, r=10, t=42, b=40),
            legend=dict(orientation="h", y=-0.22, font_size=11),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col_f:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=bars_df["duration_seconds"], name="FIB",
            marker_color=C["fib"], boxmean=True, boxpoints="outliers",
        ))
        fig_box.add_trace(go.Box(
            y=bl_df["duration_seconds"], name=f"Baseline ({bl_type})",
            marker_color=C["baseline"], boxmean=True, boxpoints="outliers",
        ))
        fig_box.update_layout(
            template=PT, height=265,
            title="Duration distribution: FIB vs baseline",
            yaxis_title="Duration (s)",
            margin=dict(l=45, r=10, t=42, b=40),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Section 4: Output tables + diagnostics ────────────────────────────────────
_sh("05 · OUTPUT TABLES & DIAGNOSTICS")
tab1, tab2, tab3 = st.tabs(["FIB Bars", "Augmented Dataset", "Parameter Guide"])

with tab1:
    st.dataframe(bars_df, use_container_width=True, height=250)
    sv = bars_df[bars_df["close_reason"].isin(["timeout", "max_events", "inactivity"])]
    if not sv.empty:
        _callout(
            f"{len(sv)} bar(s) closed by safety valve (highlighted in the price chart). "
            "If these dominate, reduce η, lower δ₀, or increase the bar timeout.",
            "warn",
        )

with tab2:
    st.dataframe(aug_df, use_container_width=True, height=250)
    st.caption(
        "Derived features: threshold_utilization · information_rate · "
        "log_duration · log_n_events · log_information_scalar · "
        "price_range · price_range_pct · vwap · is_threshold_close · bar_index"
    )

with tab3:
    st.markdown(f"""
**Active config** — model: `{model}` · info_mode: `{info_mode}` · scalarizer: `{scalarizer}`
· η = {eta} · δ₀ = {delta0}s · EWMA α = {ewma_a} · warmup = {min_wu} bars

---
#### Tuning guide

| Metric | Healthy range | Action |
|--------|--------------|--------|
| % threshold closes | > 50% | Reduce η or δ₀ if below |
| CV of events/bar | 0.2 – 0.8 | High CV → lower EWMA α; very low → raise η |
| Avg duration | ≈ δ₀ | Large gap → η too high; very short → η too low |

---
#### Understanding the parameters

**η (eta)** — scales I*, the information quantum. Think of it as "how much statistical
evidence before closing a bar". Start at 1.0. Halve it if too few bars form.

**δ₀ (delta0)** — your target bar duration in seconds. At equilibrium, average duration ≈ δ₀.

**EWMA α** — how quickly the threshold adapts to regime changes.
0.02–0.05 is conservative (stable threshold). 0.10–0.20 for rapidly-changing markets.

**Warmup bars** — the threshold seeds from the first scalar during warmup.
Use 3–5 for quick exploration, 20+ for stable long-run estimation.

---
#### Model selection

- **Gaussian** — assumes returns are N(μ, σ²). Simple and fast. Best for near-normal data.
- **GARCH** — fits a GARCH(1,1) to returns, adapting σ² after each tick.
  Best for equity and FX where volatility clusters. More robust to regime shifts.
- **Hawkes** — information comes from arrival intensity λ(t), not price level.
  Best when tick clustering matters more than price moves (order-flow, event data).

---
#### About Fisher Information

The Fisher Information Matrix I(θ) measures how much data tells us about model parameters —
it is the variance of the score ∇log f(x;θ). The **logdet** scalarizer uses log det(J + εI),
which is reparameterisation-invariant: the same bar forms regardless of parameter scaling.

A FIB bar closes when: Φ(Σᵢ ∇ℓᵢ ∇ℓᵢᵀ) ≥ I\\*  where I\\* = η · λ̄_Φ · δ₀

---
#### Python integration

```python
from fibars import build_fib_bars, augment_with_fib_features, FIBConfig

cfg = FIBConfig(model="garch", eta=1.0, delta0_seconds=60.0)
bars = build_fib_bars(df, config=cfg)          # df needs: timestamp, price
aug  = augment_with_fib_features(bars)         # adds 10 derived features
```
""")

st.divider()

# ── Section 5: Exports ────────────────────────────────────────────────────────
_sh("06 · EXPORTS")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button(
        "Download FIB Bars (CSV)", data=_csv(bars_df),
        file_name="fib_bars.csv", mime="text/csv", use_container_width=True,
    )
with d2:
    st.download_button(
        "Download Augmented (CSV)", data=_csv(aug_df),
        file_name="fib_augmented.csv", mime="text/csv", use_container_width=True,
    )
with d3:
    exp = {k: v for k, v in sm.items() if k != "reasons"}
    exp.update(sm.get("reasons", {}))
    exp["config"] = {
        "model": model, "info_mode": info_mode, "scalarizer": scalarizer,
        "eta": eta, "delta0_seconds": delta0, "ewma_alpha": ewma_a,
        "timeout_seconds": float(timeout_s), "min_warmup_events": int(min_wu),
    }
    st.download_button(
        "Download Summary (JSON)",
        data=json.dumps(exp, indent=2).encode("utf-8"),
        file_name="fib_summary.json", mime="application/json",
        use_container_width=True,
    )
with d4:
    if bl_df is not None and not bl_df.empty:
        st.download_button(
            f"Download Baseline (CSV)", data=_csv(bl_df),
            file_name=f"baseline_{bl_type}.csv", mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("Download Baseline (CSV)", disabled=True, use_container_width=True)

st.markdown(
    "<div style='text-align:center;color:#9CA3AF;font-size:.7rem;"
    "font-family:JetBrains Mono,monospace;margin-top:2rem;'>"
    "Fisher Information Bars v1.1.0 &nbsp;·&nbsp; "
    "bar closes when Σ Ψᵢ(θ̂) ≥ I*"
    "</div>",
    unsafe_allow_html=True,
)