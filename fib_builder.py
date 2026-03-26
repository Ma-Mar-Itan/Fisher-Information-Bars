"""Core FIB builder: ingests events, accumulates information, emits bars."""
from __future__ import annotations
import numpy as np
from ..config import FIBConfig
from ..events import MarketEvent
from ..models import create_model
from ..models.base import BaseLocalModel
from ..information.scalarizers import get_scalarizer, BaseScalarizer
from ..thresholds.adaptive import AdaptiveThresholdPolicy
from ..thresholds.timeout import TimeoutPolicy
from .aggregators import BarAggregator
from .outputs import FIBBar
from ..utils.math import symmetrize


class FIBBuilder:
    """
    Stateful, event-driven FIB bar builder.

    Usage
    -----
        builder = FIBBuilder(config)
        for event in stream:
            bar = builder.update(event)
            if bar is not None:
                process(bar)
        final = builder.flush()
    """

    def __init__(self, config: FIBConfig | None = None) -> None:
        self.cfg = config or FIBConfig()

        # Model instantiated via factory — config params correctly plumbed
        self._model: BaseLocalModel = create_model(self.cfg)

        self._scalarizer: BaseScalarizer = get_scalarizer(self.cfg.scalarizer)
        self._threshold_policy = AdaptiveThresholdPolicy(
            eta=self.cfg.eta,
            delta0_seconds=self.cfg.delta0_seconds,
            ewma_alpha=self.cfg.ewma_alpha,
            min_threshold=self.cfg.min_threshold,
            max_threshold=self.cfg.max_threshold,
            min_warmup_events=self.cfg.min_warmup_events,
        )
        self._timeout_policy = TimeoutPolicy(
            timeout_seconds=self.cfg.timeout_seconds,
            max_events_per_bar=self.cfg.max_events_per_bar,
            inactivity_timeout_seconds=self.cfg.inactivity_timeout_seconds,
        )
        self._agg = BarAggregator(
            price_field=self.cfg.price_field,
            volume_field=self.cfg.volume_field,
        )

        # n_params comes from model itself — no hardcoding
        self._n_params: int = self._model.n_params
        self._J: np.ndarray = np.zeros((self._n_params, self._n_params))
        self._current_scalar: float = 0.0
        self._global_event_count: int = 0
        self._last_event_time: float = 0.0

    def _reset_bar(self) -> None:
        self._J = np.zeros((self._n_params, self._n_params))
        self._current_scalar = 0.0
        self._agg.reset()

    def _compute_increment(self, event: MarketEvent) -> np.ndarray:
        if self.cfg.info_mode == "observed":
            inc = self._model.observed_information_increment(event)
        else:
            inc = self._model.expected_information_increment(event)
        # Guard against NaN/Inf in increment
        if not np.all(np.isfinite(inc)):
            return np.zeros((self._n_params, self._n_params))
        return inc

    def _scalarize(self) -> float:
        val = self._scalarizer(symmetrize(self._J), eps=self.cfg.eps_ridge)
        return val if np.isfinite(val) else 0.0

    def _emit_bar(self, close_reason: str) -> FIBBar:
        scalar = self._current_scalar
        threshold = self._threshold_policy.current_threshold()
        dur = self._agg.duration_seconds
        self._threshold_policy.update(scalar, max(dur, 1e-3))
        timeout_flag = close_reason != "threshold"
        bar = FIBBar(
            open_time=self._agg.open_time or 0.0,
            close_time=self._agg.close_time or 0.0,
            duration_seconds=dur,
            n_events=self._agg.n_events,
            open=self._agg.open,
            high=self._agg.high,
            low=self._agg.low,
            close=self._agg.close,
            sum_volume=self._agg.sum_volume,
            dollar_value=self._agg.dollar_value,
            mean_spread=self._agg.mean_spread,
            information_scalar=scalar,
            threshold_at_close=threshold,
            timeout_flag=timeout_flag,
            close_reason=close_reason,
            model_name=self.cfg.model,
            info_mode=self.cfg.info_mode,
            scalarizer_name=self.cfg.scalarizer,
            start_event_index=self._agg.start_index,
            end_event_index=self._agg.end_index,
        )
        self._reset_bar()
        return bar

    def update(self, event: MarketEvent) -> FIBBar | None:
        """Process one event. Returns a FIBBar if the bar closes, else None."""
        self._global_event_count += 1
        event.index = self._global_event_count
        prev_event_time = self._last_event_time
        self._last_event_time = event.timestamp

        # First event in bar: open it, prime the model, no increment yet
        if self._agg.n_events == 0:
            self._agg.add(event)
            self._model.update(event)
            return None

        # Compute increment at theta_{i-1} BEFORE updating model state
        increment = self._compute_increment(event)
        self._J += increment
        self._current_scalar = self._scalarize()

        # Seed threshold from first real scalar (breaks inf-deadlock at startup)
        self._threshold_policy.seed_from_scalar(self._current_scalar)

        # Update aggregator and model state
        self._agg.add(event)
        self._model.update(event)

        bar_start = self._agg.open_time or event.timestamp

        # ── Inactivity check (gap between previous and current event) ───────
        if self.cfg.inactivity_timeout_seconds is not None:
            gap = event.timestamp - prev_event_time
            if prev_event_time > 0 and gap >= self.cfg.inactivity_timeout_seconds:
                return self._emit_bar("inactivity")

        # ── Wall-clock timeout ───────────────────────────────────────────────
        if event.timestamp - bar_start >= self.cfg.timeout_seconds:
            return self._emit_bar("timeout")

        # ── Max events cap ───────────────────────────────────────────────────
        if self._agg.n_events >= self.cfg.max_events_per_bar:
            return self._emit_bar("max_events")

        # ── Information threshold ────────────────────────────────────────────
        if self._current_scalar >= self._threshold_policy.current_threshold():
            return self._emit_bar("threshold")

        return None

    def flush(self) -> FIBBar | None:
        """Force-close any open bar at end of data."""
        if self._agg.n_events > 0:
            return self._emit_bar("flush")
        return None

    def heartbeat(self, wall_time: float) -> FIBBar | None:
        """
        Poll-based timeout for live streaming with no new events.
        Call periodically (e.g. every second) from an external clock.
        Returns a bar if timeout fires, else None.
        """
        if self._agg.n_events == 0:
            return None
        bar_start = self._agg.open_time or wall_time
        if self._timeout_policy.check_heartbeat(
            bar_start_time=bar_start,
            heartbeat_time=wall_time,
            n_events_in_bar=self._agg.n_events,
            last_event_time=self._last_event_time,
        ):
            return self._emit_bar("timeout")
        return None

    # ── Live inspection ─────────────────────────────────────────────────────

    @property
    def current_scalar(self) -> float:
        return self._current_scalar

    @property
    def current_threshold(self) -> float:
        return self._threshold_policy.current_threshold()

    @property
    def n_bars_completed(self) -> int:
        return self._threshold_policy.n_bars_completed

    def model_state(self) -> dict:
        return self._model.state_dict()

    def threshold_state(self) -> dict:
        return self._threshold_policy.state_dict()
