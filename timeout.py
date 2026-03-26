"""Timeout / safety-valve policy.

A bar is force-closed (timeout_flag=True) when ANY condition fires:
  1. Elapsed wall-clock time   >= timeout_seconds
  2. Event count in bar        >= max_events_per_bar
  3. Inactivity gap            >= inactivity_timeout_seconds  (optional)

FIBBuilder passes last_event_time separately from current_time, so condition 3
is ONLY meaningful when called from an external heartbeat/clock poll.
During normal streaming (last_event_time == current_time), condition 3 never
fires mid-stream — it fires when the *next* event arrives after a long gap.
"""
from __future__ import annotations
from typing import Optional


class TimeoutPolicy:
    """
    Stateless evaluator — call should_timeout() on every event.

    Parameters
    ----------
    timeout_seconds : float
    max_events_per_bar : int
    inactivity_timeout_seconds : float | None
        Close if (current_time - last_event_time) >= this value.
        In batch mode, pass last_event_time from the previous event
        and current_time from the current event to detect gaps.
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        max_events_per_bar: int = 10_000,
        inactivity_timeout_seconds: Optional[float] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_events_per_bar < 1:
            raise ValueError(f"max_events_per_bar must be >= 1, got {max_events_per_bar}")
        if inactivity_timeout_seconds is not None and inactivity_timeout_seconds <= 0:
            raise ValueError(f"inactivity_timeout_seconds must be positive")
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
        bar_start_time : float     — timestamp of first event in current bar
        current_time : float       — timestamp of event just processed
        n_events_in_bar : int      — events accumulated so far (including current)
        last_event_time : float    — timestamp of the PREVIOUS event
                                     (set to current_time for streaming mode)
        """
        # Condition 1: wall-clock elapsed
        if current_time - bar_start_time >= self._timeout:
            return True

        # Condition 2: event count cap
        if n_events_in_bar >= self._max_events:
            return True

        # Condition 3: inactivity gap (gap between last and current event)
        if self._inactivity is not None:
            gap = current_time - last_event_time
            if gap >= self._inactivity:
                return True

        return False

    def check_heartbeat(
        self,
        bar_start_time: float,
        heartbeat_time: float,
        n_events_in_bar: int,
        last_event_time: float,
    ) -> bool:
        """
        Poll-based timeout check for live streaming with no new events.
        Call periodically when no event has arrived.
        """
        # Wall-clock timeout
        if heartbeat_time - bar_start_time >= self._timeout:
            return True
        # Inactivity
        if self._inactivity is not None:
            if heartbeat_time - last_event_time >= self._inactivity:
                return True
        return False
