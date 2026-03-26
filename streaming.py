"""StreamingBuilder: push-based interface for real-time tick feeds."""
from __future__ import annotations
from typing import Callable, Optional
from ..config import FIBConfig
from ..bars.fib_builder import FIBBuilder
from ..bars.outputs import FIBBar
from ..events import MarketEvent


class StreamingBuilder:
    """
    Event-by-event FIB builder for live / streaming contexts.

    Usage
    -----
        builder = StreamingBuilder(config=FIBConfig(model="gaussian"))
        for raw in tick_feed:
            bar = builder.push(MarketEvent(timestamp=raw.ts, price=raw.price))
            if bar is not None:
                handle(bar)
        builder.flush()
    """

    def __init__(
        self,
        config: FIBConfig | None = None,
        on_bar: Optional[Callable[[FIBBar], None]] = None,
    ) -> None:
        self._builder = FIBBuilder(config)
        self._on_bar = on_bar
        self._completed: list[FIBBar] = []

    def push(self, event: MarketEvent) -> FIBBar | None:
        bar = self._builder.update(event)
        if bar is not None:
            self._completed.append(bar)
            if self._on_bar is not None:
                self._on_bar(bar)
        return bar

    def flush(self) -> FIBBar | None:
        bar = self._builder.flush()
        if bar is not None:
            self._completed.append(bar)
            if self._on_bar is not None:
                self._on_bar(bar)
        return bar

    def heartbeat(self, wall_time: float) -> FIBBar | None:
        """Poll for timeout when no events are arriving."""
        bar = self._builder.heartbeat(wall_time)
        if bar is not None:
            self._completed.append(bar)
            if self._on_bar is not None:
                self._on_bar(bar)
        return bar

    @property
    def completed_bars(self) -> list[FIBBar]:
        return list(self._completed)

    @property
    def current_scalar(self) -> float:
        return self._builder.current_scalar

    @property
    def current_threshold(self) -> float:
        return self._builder.current_threshold

    @property
    def n_bars_completed(self) -> int:
        return self._builder.n_bars_completed

    def reset(self, config: FIBConfig | None = None) -> None:
        self._builder = FIBBuilder(config or self._builder.cfg)
        self._completed = []
