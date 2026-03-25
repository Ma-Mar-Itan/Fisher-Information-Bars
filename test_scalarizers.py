"""Unit tests for scalarizers."""
import numpy as np
import pytest
from fibars.information.scalarizers import (
    LogdetScalarizer, TraceScalarizer, FrobeniusScalarizer, get_scalarizer
)


def spd_matrix(n=2, scale=1.0):
    """Random symmetric positive definite matrix."""
    rng = np.random.default_rng(42)
    A = rng.normal(size=(n, n)) * scale
    return A @ A.T + np.eye(n) * 0.1


class TestLogdetScalarizer:

    def test_zero_matrix_finite(self):
        s = LogdetScalarizer()
        val = s(np.zeros((2, 2)), eps=1e-6)
        assert np.isfinite(val)

    def test_identity_close_to_zero(self):
        s = LogdetScalarizer()
        val = s(np.eye(3), eps=0.0)
        assert np.isclose(val, 0.0, atol=1e-8)

    def test_larger_matrix_gives_larger_value(self):
        s = LogdetScalarizer()
        J_small = np.eye(2)
        J_large = 10.0 * np.eye(2)
        assert s(J_large) > s(J_small)

    def test_symmetric_input(self):
        s = LogdetScalarizer()
        J = spd_matrix(3)
        val = s(J)
        assert np.isfinite(val)

    def test_non_psd_stable(self):
        s = LogdetScalarizer()
        J = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
        val = s(J, eps=1e-4)
        assert np.isfinite(val)


class TestTraceScalarizer:

    def test_zero_matrix(self):
        s = TraceScalarizer()
        val = s(np.zeros((2, 2)), eps=1e-6)
        assert val > 0  # eps * d contributes

    def test_proportional_to_diagonal_sum(self):
        s = TraceScalarizer()
        J = np.diag([1.0, 2.0, 3.0])
        val = s(J, eps=0.0)
        assert np.isclose(val, 6.0, atol=1e-8)

    def test_scales_linearly(self):
        s = TraceScalarizer()
        J = np.eye(2)
        assert np.isclose(s(2.0 * J, eps=0.0), 2.0 * s(J, eps=0.0), rtol=1e-6)


class TestFrobeniusScalarizer:

    def test_zero_matrix(self):
        s = FrobeniusScalarizer()
        val = s(np.zeros((2, 2)), eps=1e-6)
        assert np.isclose(val, 1e-6)

    def test_identity(self):
        s = FrobeniusScalarizer()
        # ||I_2||_F = sqrt(2)
        val = s(np.eye(2), eps=0.0)
        assert np.isclose(val, np.sqrt(2.0), atol=1e-8)

    def test_positive_for_nonzero(self):
        s = FrobeniusScalarizer()
        J = spd_matrix(3)
        assert s(J) > 0


class TestGetScalarizer:

    def test_valid_names(self):
        for name in ["logdet", "trace", "frobenius"]:
            s = get_scalarizer(name)
            assert callable(s)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown scalarizer"):
            get_scalarizer("entropy")

    def test_returned_scalarizer_callable_with_matrix(self):
        for name in ["logdet", "trace", "frobenius"]:
            s = get_scalarizer(name)
            val = s(np.eye(2))
            assert np.isfinite(val)
