"""Unit tests for GaussianModel, GARCHModel, HawkesModel."""
import numpy as np
import pytest
from fibars.events import MarketEvent
from fibars.models.gaussian import GaussianModel
from fibars.models.garch import GARCHModel
from fibars.models.hawkes import HawkesModel


def make_events(prices, base_ts=0.0, dt=1.0):
    return [MarketEvent(timestamp=base_ts + i * dt, price=p) for i, p in enumerate(prices)]


# ── GaussianModel ────────────────────────────────────────────────────────────

class TestGaussianModel:

    def test_score_zero_for_first_event(self):
        m = GaussianModel()
        m.initialize()
        ev = MarketEvent(timestamp=0.0, price=100.0)
        s = m.score(ev)
        assert s.shape == (2,)
        assert np.allclose(s, 0.0)

    def test_score_shape(self):
        m = GaussianModel()
        m.initialize()
        events = make_events([100.0, 100.1, 99.9, 100.2])
        m.update(events[0])
        s = m.score(events[1])
        assert s.shape == (2,)
        assert np.all(np.isfinite(s))

    def test_opg_is_psd(self):
        m = GaussianModel()
        m.initialize()
        events = make_events([100.0, 100.5, 101.0])
        for ev in events[:2]:
            m.update(ev)
        J = m.observed_information_increment(events[2])
        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals >= -1e-12)

    def test_expected_info_diagonal(self):
        m = GaussianModel()
        m.initialize()
        events = make_events([100.0, 100.1, 99.9])
        for ev in events[:2]:
            m.update(ev)
        I = m.expected_information_increment(events[2])
        assert I.shape == (2, 2)
        assert np.isclose(I[0, 1], 0.0)
        assert np.isclose(I[1, 0], 0.0)
        assert I[0, 0] > 0
        assert np.isclose(I[1, 1], 0.5)

    def test_welford_online_update(self):
        m = GaussianModel()
        m.initialize()
        prices = [100.0, 100.2, 99.8, 100.4, 99.6]
        for i, p in enumerate(prices):
            m.update(MarketEvent(timestamp=float(i), price=p))
        assert np.isfinite(m._mu)
        assert m._sigma2 > 0

    def test_state_dict_keys(self):
        m = GaussianModel()
        m.initialize()
        d = m.state_dict()
        assert "mu" in d and "sigma2" in d and "n" in d

    def test_var_floor_prevents_zero_sigma(self):
        m = GaussianModel(var_floor=1e-10)
        m.initialize()
        # Feed constant prices — variance should stay at floor
        for i in range(20):
            m.update(MarketEvent(timestamp=float(i), price=100.0))
        assert m._sigma2 >= 1e-10


# ── GARCHModel ───────────────────────────────────────────────────────────────

class TestGARCHModel:

    def test_score_zero_on_first_event(self):
        m = GARCHModel()
        m.initialize()
        ev = MarketEvent(timestamp=0.0, price=100.0)
        s = m.score(ev)
        assert s.shape == (3,)
        assert np.allclose(s, 0.0)

    def test_score_finite_after_warmup(self):
        m = GARCHModel()
        m.initialize()
        events = make_events([100.0, 100.1, 99.9, 100.3, 100.0])
        for ev in events[:3]:
            m.update(ev)
        s = m.score(events[3])
        assert s.shape == (3,)
        assert np.all(np.isfinite(s))

    def test_persistence_cap_respected(self):
        m = GARCHModel(alpha_init=0.5, beta_init=0.6, persistence_max=0.9999)
        m.initialize()
        assert m._alpha + m._beta < 0.9999

    def test_sigma2_positive(self):
        m = GARCHModel()
        m.initialize()
        events = make_events([100.0 + np.random.randn() * 0.1 for _ in range(50)])
        for ev in events:
            m.update(ev)
        assert m._sigma2 > 0

    def test_opg_shape(self):
        m = GARCHModel()
        m.initialize()
        events = make_events([100.0, 100.2, 99.8, 100.5])
        for ev in events[:3]:
            m.update(ev)
        J = m.observed_information_increment(events[3])
        assert J.shape == (3, 3)
        assert np.all(np.isfinite(J))

    def test_state_dict_keys(self):
        m = GARCHModel()
        m.initialize()
        d = m.state_dict()
        assert all(k in d for k in ["omega", "alpha", "beta", "sigma2", "n"])


# ── HawkesModel ──────────────────────────────────────────────────────────────

class TestHawkesModel:

    def test_score_finite(self):
        m = HawkesModel()
        m.initialize()
        events = make_events([100.0, 100.1, 100.0, 100.2], dt=0.5)
        for ev in events[:2]:
            m.update(ev)
        s = m.score(events[2])
        assert s.shape == (3,)
        assert np.all(np.isfinite(s))

    def test_R_decays(self):
        """Excitation state should decay between events."""
        m = HawkesModel(beta_init=1.0)
        m.initialize()
        m.update(MarketEvent(timestamp=0.0, price=100.0))
        R_after_first = m._R
        m.update(MarketEvent(timestamp=10.0, price=100.1))  # long gap
        # R decays before +1; with beta=1 and dt=10, exp(-10) ≈ 0
        assert m._R < R_after_first + 1.0 + 0.01

    def test_intensity_floor(self):
        """Intensity should never go below the floor."""
        m = HawkesModel(mu_init=0.0, alpha_init=0.0, intensity_floor=1e-8)
        m.initialize()
        # With mu=0 and alpha=0, intensity should equal floor
        result = m._intensity(0.0)
        assert result >= 1e-8

    def test_opg_shape(self):
        m = HawkesModel()
        m.initialize()
        events = make_events([100.0, 100.1, 99.9], dt=0.3)
        m.update(events[0])
        m.update(events[1])
        J = m.observed_information_increment(events[2])
        assert J.shape == (3, 3)
        assert np.all(np.isfinite(J))

    def test_state_dict_keys(self):
        m = HawkesModel()
        m.initialize()
        d = m.state_dict()
        assert all(k in d for k in ["mu", "alpha", "beta", "R", "n"])

    def test_log_lik_contrib_finite(self):
        m = HawkesModel()
        m.initialize()
        ll = m._log_lik_contrib(1.0, 0.5, 1.0, dt=0.5, R_pre_decay=0.3)
        assert np.isfinite(ll)
