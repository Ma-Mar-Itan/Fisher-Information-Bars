"""Tests for AdaptiveThresholdPolicy and TimeoutPolicy."""
import pytest
from fibars.thresholds.adaptive import AdaptiveThresholdPolicy
from fibars.thresholds.timeout import TimeoutPolicy


class TestAdaptiveThresholdPolicy:

    def test_initial_threshold_is_inf(self):
        p = AdaptiveThresholdPolicy()
        assert p.current_threshold() == float("inf")

    def test_seed_breaks_deadlock(self):
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
        p.seed_from_scalar(100.0)
        assert p.current_threshold() == first

    def test_update_changes_threshold(self):
        p = AdaptiveThresholdPolicy(min_warmup_events=1)
        p.seed_from_scalar(1.0)
        p.update(bar_scalar=10.0, bar_duration_seconds=60.0)
        assert p.current_threshold() > 0.0
        assert p.n_bars_completed == 1

    def test_ewma_converges(self):
        p = AdaptiveThresholdPolicy(eta=1.0, delta0_seconds=60.0,
                                    ewma_alpha=0.1, min_warmup_events=3)
        p.seed_from_scalar(1.0)
        for _ in range(60):
            p.update(bar_scalar=10.0, bar_duration_seconds=60.0)
        # rate=10/60, threshold = 1 * (10/60) * 60 = 10
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
        assert {"n_bars_completed", "ewma_rate", "last_scalar", "threshold", "seeded"} <= set(d)


class TestTimeoutPolicy:

    def test_no_timeout_within_limits(self):
        p = TimeoutPolicy(timeout_seconds=60.0, max_events_per_bar=100)
        assert not p.should_timeout(0.0, 30.0, 50, 30.0)

    def test_wall_clock_triggers(self):
        p = TimeoutPolicy(timeout_seconds=60.0)
        assert p.should_timeout(0.0, 60.0, 1, 60.0)

    def test_wall_clock_boundary_exact(self):
        p = TimeoutPolicy(timeout_seconds=60.0)
        assert p.should_timeout(0.0, 60.0, 1, 60.0)
        assert not p.should_timeout(0.0, 59.99, 1, 59.99)

    def test_max_events_triggers(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, max_events_per_bar=100)
        assert p.should_timeout(0.0, 1.0, 100, 1.0)

    def test_inactivity_triggers(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, inactivity_timeout_seconds=30.0)
        # current_time=100, last_event=60 → gap=40 >= 30
        assert p.should_timeout(0.0, 100.0, 5, 60.0)

    def test_inactivity_not_triggered_below_threshold(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, inactivity_timeout_seconds=30.0)
        # gap = 10 < 30
        assert not p.should_timeout(0.0, 100.0, 5, 90.0)

    def test_inactivity_none_by_default(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, max_events_per_bar=9999)
        # 100s since last event but no inactivity rule
        assert not p.should_timeout(0.0, 100.0, 5, 0.0)

    def test_heartbeat_fires_on_wall_clock(self):
        p = TimeoutPolicy(timeout_seconds=30.0)
        assert p.check_heartbeat(0.0, 35.0, 5, 10.0)

    def test_heartbeat_fires_on_inactivity(self):
        p = TimeoutPolicy(timeout_seconds=9999.0, inactivity_timeout_seconds=10.0)
        assert p.check_heartbeat(0.0, 50.0, 5, 30.0)  # 50-30=20 >= 10

    def test_invalid_timeout_raises(self):
        with pytest.raises(ValueError):
            TimeoutPolicy(timeout_seconds=-1.0)

    def test_invalid_max_events_raises(self):
        with pytest.raises(ValueError):
            TimeoutPolicy(max_events_per_bar=0)
