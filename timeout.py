"""Timeout / safety-valve policy.

A bar is force-closed (timeout_flag=True) when ANY of three conditions fires:
  1. Elapsed wall-clock time >= timeout_seconds
  2. Event count in bar >= max_events_per_bar
  3. Seconds since last event >= inactivity_timeout_seconds  (optional)

This prevents "bar starvation" in dead markets where the information threshold
is never reached.
"""
from __future__ import annotations
from typing import Optional


class TimeoutPolicy:
    """
    Stateless evaluator — call should_timeout() on every event.

    Parameters
    ----------
    timeout_seconds : float
        Maximum wall-clock duration of a bar.
    max_events_per_bar : int
        Hard cap on the number of events a single bar may contain.
    inactivity_timeout_seconds : float | None
        If set, close the bar when no new event has arrived for this long.
        Measured as (current_time - last_event_time).  Useful for illiquid
        assets that may go silent for extended periods mid-bar.
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        max_events_per_bar: int = 10_000,
        inactivity_timeout_seconds: Optional[float] = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_events = max_events_per_bar
        self._inactivity = inactivity_timeout_seconds

    def should_timeout(
        self,
        bar_start_time: float,
        current_time: float,
        n_events_in_bar: int,
        last_event_time: float,
    ) -> bool:
        """
        Return True if any timeout condition is met.

        Parameters
        ----------
        bar_start_time : float
            Timestamp of the first event in the current bar.
        current_time : float
            Timestamp of the event just processed.
        n_events_in_bar : int
            Number of events accumulated in the current bar so far.
        last_event_time : float
            Timestamp of the most recent event (same as current_time during
            normal processing; may differ if called externally).
        """
        # Condition 1: wall-clock elapsed
        if current_time - bar_start_time >= self._timeout:
            return True

        # Condition 2: event count cap
        if n_events_in_bar >= self._max_events:
            return True

        # Condition 3: inactivity (checked externally — not triggered here
        #   during normal update() since current_time == last_event_time;
        #   relevant for a polling / heartbeat scenario)
        if self._inactivity is not None:
            if current_time - last_event_time >= self._inactivity:
                return True

        return False
