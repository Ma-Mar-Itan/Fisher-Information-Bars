"""StreamingBuilder: thin wrapper around FIBBuilder for real-time tick feeds.

Exposes a push() / flush() interface that matches the README quickstart.
Also supports an optional callback for zero-latency bar handling.
"""
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
    builder = StreamingBuilder(config=FIBConfig(model="gaussian", eta=1.0))
    for raw in tick_feed:
        event = MarketEvent(timestamp=raw.ts, price=raw.price, size=raw.qty)
        bar = builder.push(event)
        if bar is not None:
            handle_closed_bar(bar)
    final = builder.flush()

    Alternatively, register a callback:
    builder = StreamingBuilder(config, on_bar=handle_closed_bar)
    for raw in tick_feed:
        builder.push(MarketEvent(...))  # callback fires automatically
    builder.flush()
    """

    def __init__(
        self,
        config: FIBConfig | None = None,
        on_bar: Optional[Callable[[FIBBar], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : FIBConfig | None
            Full configuration.  Defaults to FIBConfig() (Gaussian, logdet).
        on_bar : callable | None
            If provided, called immediately whenever a bar closes.
            The return value of push() will still be the bar (or None).
        """
        self._builder = FIBBuilder(config)
        self._on_bar = on_bar
        self._completed: list[FIBBar] = []

    def push(self, event: MarketEvent) -> FIBBar | None:
        """
        Ingest one market event.

        Returns the completed FIBBar if one closed, else None.
        Also fires the on_bar callback if registered.
        """
        bar = self._builder.update(event)
        if bar is not None:
            self._completed.append(bar)
            if self._on_bar is not None:
                self._on_bar(bar)
        return bar

    def flush(self) -> FIBBar | None:
        """
        Force-close any open bar (call at end of data or session).

        Returns the final bar if any events were open, else None.
        """
        bar = self._builder.flush()
        if bar is not None:
            self._completed.append(bar)
            if self._on_bar is not None:
                self._on_bar(bar)
        return bar

    # ── Inspection ──────────────────────────────────────────────────────────

    @property
    def completed_bars(self) -> list[FIBBar]:
        """All bars completed so far in this session."""
        return list(self._completed)

    @property
    def current_scalar(self) -> float:
        """Live scalarised information accumulated in the open bar."""
        return self._builder.current_scalar

    @property
    def current_threshold(self) -> float:
        """Current Information Quantum I*."""
        return self._builder.current_threshold

    @property
    def n_bars_completed(self) -> int:
        return self._builder.n_bars_completed

    def reset(self, config: FIBConfig | None = None) -> None:
        """
        Hard reset — start a fresh session, optionally with new config.
        Clears completed bar history.
        """
        self._builder = FIBBuilder(config or self._builder.cfg)
        self._completed = []
