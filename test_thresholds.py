"""Unit tests for AdaptiveThresholdPolicy and TimeoutPolicy."""
import pytest
from fibars.thresholds.adaptive import AdaptiveThresholdPolicy
from fibars.thresholds.timeout import TimeoutPolicy


# ── AdaptiveThresholdPolicy ───────────────────────────────────────────────────

class TestAdaptiveThresholdPolicy:

    def test_initial_threshold_is_inf(self):
        p = AdaptiveThresholdPolicy()
        assert p.current_threshold() == float("inf")

    def test_seed_from_scalar_breaks_deadlock(self):
        p = AdaptiveThresholdPolicy()
        p.seed_from_scalar(5.0)
        assert p.current_threshold() < float("inf")
        assert p.current_threshold() > 0.0

    def test_seed_ignored_if_zero(self):
        p = AdaptiveThresholdPolicy()
        p.seed_from_scalar(0.0)
        assert p.current_threshold() == float("inf")

    def test_seed_only_fires_once(self):
        p = AdaptiveThresholdPolicy()
        p.seed_from_scalar(5.0)
        first = p.current_threshold()
        p.seed_from_scalar(100.0)  # should be ignored
        assert p.current_threshold() == first

    def test_update_changes_threshold(self):
        p = AdaptiveThresholdPolicy(min_warmup_events=1)
        p.seed_from_scalar(1.0)
        p.update(bar_scalar=10.0, bar_duration_seconds=60.0)
        # After warmup threshold should be based on EWMA rate
        assert p.current_threshold() > 0.0
        assert p.n_bars_completed == 1

    def test_ewma_converges(self):
        """After many bars, threshold should stabilize."""
        p = AdaptiveThresholdPolicy(eta=1.0, delta0_seconds=60.0,
                                    ewma_alpha=0.1, min_warmup_events=3)
        p.seed_from_scalar(1.0)
        for _ in range(50):
            p.update(bar_scalar=10.0, bar_duration_seconds=60.0)
        # With scalar=10 and dur=60, rate=10/60, threshold=eta*rate*delta0=10
        assert abs(p.current_threshold() - 10.0) < 1.0

    def test_min_threshold_floor(self):
        p = AdaptiveThresholdPolicy(min_threshold=100.0, min_warmup_events=1)
        p.seed_from_scalar(0.001)
        p.update(bar_scalar=0.001, bar_duration_seconds=1.0)
        assert p.current_threshold() >= 100.0

    def test_max_threshold_ceiling(self):
        p = AdaptiveThresholdPolicy(max_threshold=5.0, min_warmup_events=1)
        p.seed_from_scalar(1.0)
        p.update(bar_scalar=1000.0, bar_duration_seconds=1.0)
        assert p.current_threshold() <= 5.0

    def test_state_dict_complete(self):
        p = AdaptiveThresholdPolicy()
        d = p.state_dict()
        assert all(k in d for k in ["n_bars_completed", "ewma_rate",
                                     "last_scalar", "threshold", "seeded"])


# ── TimeoutPolicy ─────────────────────────────────────────────────────────────

class TestTimeoutPolicy:

    def test_no_timeout_within_limits(self):
        p = TimeoutPolicy(timeout_seconds=60.0, max_events_per_bar=100)
        assert not p.should_timeout(
            bar_start_time=0.0, current_time=30.0,
            n_events_in_bar=50, last_event_time=30.0
        )

    def test_wall_clock_timeout(self):
        p = TimeoutPolicy(timeout_seconds=60.0)
        assert p.should_timeout(
            bar_start_time=0.0, current_time=60.0,
            n_events_in_bar=1, last_event_time=60.0
        )

    def test_wall_clock_boundary(self):
        p = TimeoutPolicy(timeout_seconds=60.0)
        # Exactly at boundary should trigger
        assert p.should_timeout(
            bar_start_time=0.0, current_time=60.0,
            n_events_in_bar=1, last_event_time=60.0
        )
        # One second before should not
        assert not p.should_timeout(
            bar_start_time=0.0, current_time=59.99,
            n_events_in_bar=1, last_event_time=59.99
        )

    def test_max_events_timeout(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, max_events_per_bar=100)
        assert p.should_timeout(
            bar_start_time=0.0, current_time=1.0,
            n_events_in_bar=100, last_event_time=1.0
        )

    def test_inactivity_timeout(self):
        p = TimeoutPolicy(timeout_seconds=9999.0,
                          inactivity_timeout_seconds=30.0)
        assert p.should_timeout(
            bar_start_time=0.0, current_time=100.0,
            n_events_in_bar=5, last_event_time=60.0  # 40s gap
        )

    def test_inactivity_not_triggered_below_threshold(self):
        p = TimeoutPolicy(timeout_seconds=9999.0,
                          inactivity_timeout_seconds=30.0)
        assert not p.should_timeout(
            bar_start_time=0.0, current_time=100.0,
            n_events_in_bar=5, last_event_time=90.0  # 10s gap
        )

    def test_inactivity_none_by_default(self):
        """With no inactivity setting, only wall-clock and event-count matter."""
        p = TimeoutPolicy(timeout_seconds=9999.0, max_events_per_bar=9999)
        assert not p.should_timeout(
            bar_start_time=0.0, current_time=100.0,
            n_events_in_bar=5, last_event_time=0.0  # 100s since last event
        )
