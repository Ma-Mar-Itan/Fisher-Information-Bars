"""Unit tests for GaussianModel, GARCHModel, HawkesModel."""
import numpy as np
import pytest
from fibars.events import MarketEvent
from fibars.models.gaussian import GaussianModel
from fibars.models.garch import GARCHModel
from fibars.models.hawkes import HawkesModel
from fibars.models import create_model
from fibars.config import FIBConfig


def make_events(prices, base_ts=0.0, dt=1.0):
    return [MarketEvent(timestamp=base_ts + i * dt, price=p) for i, p in enumerate(prices)]


# ── GaussianModel ────────────────────────────────────────────────────────────

class TestGaussianModel:

    def test_n_params(self):
        assert GaussianModel().n_params == 2

    def test_score_zero_before_first_update(self):
        m = GaussianModel()
        m.initialize()
        s = m.score(MarketEvent(timestamp=0.0, price=100.0))
        assert s.shape == (2,)
        assert np.allclose(s, 0.0)

    def test_score_finite_after_warmup(self):
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
        assert I[0, 0] > 0
        assert I[1, 1] > 0

    def test_var_floor_prevents_zero_sigma(self):
        m = GaussianModel(var_floor=1e-10)
        m.initialize()
        for i in range(20):
            m.update(MarketEvent(timestamp=float(i), price=100.0))
        assert m._sigma2 >= 1e-10

    def test_var_floor_config_plumbing(self):
        cfg = FIBConfig(model="gaussian", var_floor=1e-8)
        model = create_model(cfg)
        assert model._var_floor == 1e-8

    def test_state_dict_keys(self):
        m = GaussianModel()
        m.initialize()
        d = m.state_dict()
        assert {"mu", "sigma2", "n"} <= set(d)

    def test_welford_variance_converges(self):
        rng = np.random.default_rng(0)
        prices = 100.0 + rng.normal(0, 2.0, 1000)
        m = GaussianModel()
        m.initialize()
        for i, p in enumerate(prices):
            m.update(MarketEvent(timestamp=float(i), price=p))
        assert abs(m._sigma2 - 4.0) < 0.5  # variance ≈ 4


# ── GARCHModel ───────────────────────────────────────────────────────────────

class TestGARCHModel:

    def test_n_params(self):
        assert GARCHModel().n_params == 3

    def test_score_zero_on_first_event(self):
        m = GARCHModel()
        m.initialize()
        s = m.score(MarketEvent(timestamp=0.0, price=100.0))
        assert s.shape == (3,)
        assert np.allclose(s, 0.0)

    def test_score_finite_after_warmup(self):
        m = GARCHModel()
        m.initialize()
        events = make_events([100.0, 100.1, 99.9, 100.3, 100.0])
        for ev in events[:3]:
            m.update(ev)
        s = m.score(events[3])
        assert np.all(np.isfinite(s))

    def test_persistence_cap_at_init(self):
        m = GARCHModel(alpha_init=0.5, beta_init=0.6, persistence_max=0.9999)
        assert m._alpha + m._beta < 0.9999

    def test_persistence_cap_config_plumbing(self):
        cfg = FIBConfig(model="garch", garch_persistence_max=0.95)
        model = create_model(cfg)
        assert model._alpha + model._beta < 0.95

    def test_sigma2_positive(self):
        m = GARCHModel()
        m.initialize()
        rng = np.random.default_rng(1)
        events = make_events(100.0 + rng.normal(0, 0.1, 50))
        for ev in events:
            m.update(ev)
        assert m._sigma2 > 0

    def test_opg_shape_and_finite(self):
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
        assert {"omega", "alpha", "beta", "sigma2", "n"} <= set(d)


# ── HawkesModel ──────────────────────────────────────────────────────────────

class TestHawkesModel:

    def test_n_params(self):
        assert HawkesModel().n_params == 3

    def test_score_finite(self):
        m = HawkesModel()
        m.initialize()
        events = make_events([100.0, 100.1, 100.0, 100.2], dt=0.5)
        for ev in events[:2]:
            m.update(ev)
        s = m.score(events[2])
        assert s.shape == (3,)
        assert np.all(np.isfinite(s))

    def test_R_decays_over_long_gap(self):
        m = HawkesModel(beta_init=1.0)
        m.initialize()
        m.update(MarketEvent(timestamp=0.0, price=100.0))
        m.update(MarketEvent(timestamp=0.1, price=100.1))
        R_early = m._R
        m.update(MarketEvent(timestamp=100.0, price=100.2))  # huge gap
        assert m._R < R_early  # R should have decayed

    def test_intensity_floor_config_plumbing(self):
        cfg = FIBConfig(model="hawkes", hawkes_intensity_floor=1e-5)
        model = create_model(cfg)
        assert model._intensity_floor == 1e-5

    def test_intensity_floor_respected(self):
        m = HawkesModel(mu_init=0.0, alpha_init=0.0, intensity_floor=1e-6)
        m.initialize()
        result = m._intensity(0.0)
        assert result >= 1e-6

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
        assert {"mu", "alpha", "beta", "R", "n"} <= set(d)


# ── Model registry ───────────────────────────────────────────────────────────

class TestModelRegistry:

    @pytest.mark.parametrize("model_name", ["gaussian", "garch", "hawkes"])
    def test_create_model_returns_correct_type(self, model_name):
        cfg = FIBConfig(model=model_name)
        model = create_model(cfg)
        assert model.n_params >= 2

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            from fibars.models import create_model as cm
            from fibars.config import FIBConfig as FC
            cfg = FC.__new__(FC)
            cfg.model = "unknown"
            cfg.var_floor = 1e-12
            cfg.garch_persistence_max = 0.9999
            cfg.hawkes_intensity_floor = 1e-8
            cm(cfg)

    def test_var_floor_reaches_gaussian(self):
        cfg = FIBConfig(model="gaussian", var_floor=1e-5)
        m = create_model(cfg)
        assert m._var_floor == 1e-5

    def test_persistence_max_reaches_garch(self):
        cfg = FIBConfig(model="garch", garch_persistence_max=0.80)
        m = create_model(cfg)
        assert m._persistence_max == 0.80

    def test_intensity_floor_reaches_hawkes(self):
        cfg = FIBConfig(model="hawkes", hawkes_intensity_floor=1e-4)
        m = create_model(cfg)
        assert m._intensity_floor == 1e-4
