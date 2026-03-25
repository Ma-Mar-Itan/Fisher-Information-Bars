"""Core FIB builder: ingests events, accumulates information, emits bars."""
from __future__ import annotations
import numpy as np
from ..config import FIBConfig
from ..events import MarketEvent
from ..models import REGISTRY as MODEL_REGISTRY
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

    Usage:
        builder = FIBBuilder(config)
        for event in stream:
            bar = builder.update(event)
            if bar is not None:
                process(bar)
        # Flush any open bar at end of data:
        final = builder.flush()
    """

    def __init__(self, config: FIBConfig | None = None) -> None:
        self.cfg = config or FIBConfig()
        self._model: BaseLocalModel = MODEL_REGISTRY[self.cfg.model]()
        self._model.initialize()
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
        self._n_params: int = self._get_n_params()
        self._J: np.ndarray = np.zeros((self._n_params, self._n_params))
        self._current_scalar: float = 0.0
        self._global_event_count: int = 0

    def _get_n_params(self) -> int:
        mapping = {"gaussian": 2, "garch": 3, "hawkes": 3}
        return mapping.get(self.cfg.model, 2)

    def _reset_bar(self) -> None:
        self._J = np.zeros((self._n_params, self._n_params))
        self._current_scalar = 0.0
        self._agg.reset()

    def _compute_increment(self, event: MarketEvent) -> np.ndarray:
        if self.cfg.info_mode == "observed":
            return self._model.observed_information_increment(event)
        else:
            return self._model.expected_information_increment(event)

    def _scalarize(self) -> float:
        J_sym = symmetrize(self._J)
        return self._scalarizer(J_sym, eps=self.cfg.eps_ridge)

    def _emit_bar(self, timeout_flag: bool) -> FIBBar:
        scalar = self._current_scalar
        threshold = self._threshold_policy.current_threshold()
        dur = self._agg.duration_seconds
        self._threshold_policy.update(scalar, dur)
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

        # Add to OHLCV aggregator first (for open_time)
        if self._agg.n_events == 0:
            self._agg.add(event)
            self._model.update(event)
            return None

        # Compute increment BEFORE updating model state
        increment = self._compute_increment(event)
        self._J += increment
        self._current_scalar = self._scalarize()

        # Update aggregator and model
        self._agg.add(event)
        self._model.update(event)

        # Check timeout
        bar_start = self._agg.open_time or event.timestamp
        if self._timeout_policy.should_timeout(
            bar_start_time=bar_start,
            current_time=event.timestamp,
            n_events_in_bar=self._agg.n_events,
            last_event_time=event.timestamp,
        ):
            return self._emit_bar(timeout_flag=True)

        # Check information threshold
        threshold = self._threshold_policy.current_threshold()
        if self._current_scalar >= threshold:
            return self._emit_bar(timeout_flag=False)

        return None

    def flush(self) -> FIBBar | None:
        """Force-close any open bar (call at end of data)."""
        if self._agg.n_events > 0:
            return self._emit_bar(timeout_flag=True)
        return None
