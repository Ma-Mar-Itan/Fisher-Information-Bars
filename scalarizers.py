"""Scalarizers: collapse the information matrix J to a real-valued metric."""
from __future__ import annotations
import numpy as np
from ..utils.math import ridge_logdet, symmetrize


class BaseScalarizer:
    def __call__(self, J: np.ndarray, eps: float = 1e-6) -> float:
        raise NotImplementedError


class LogdetScalarizer(BaseScalarizer):
    """
    Phi_D(J) = log det(J + eps*I)

    Reparameterisation-invariant. Preferred default.
    """
    def __call__(self, J: np.ndarray, eps: float = 1e-6) -> float:
        return ridge_logdet(symmetrize(J), eps=eps)


class TraceScalarizer(BaseScalarizer):
    """
    Phi_T(J) = tr(J + eps*I)

    Simple, fast. Sensitive to scale but not reparameterisation-invariant.
    """
    def __call__(self, J: np.ndarray, eps: float = 1e-6) -> float:
        return float(np.trace(J) + eps * J.shape[0])


class FrobeniusScalarizer(BaseScalarizer):
    """
    Phi_F(J) = ||J||_F + eps

    Frobenius norm. Captures off-diagonal covariance structure.
    """
    def __call__(self, J: np.ndarray, eps: float = 1e-6) -> float:
        val = float(np.linalg.norm(J, "fro"))
        return max(val, eps)


_SCALARIZERS: dict[str, BaseScalarizer] = {
    "logdet": LogdetScalarizer(),
    "trace": TraceScalarizer(),
    "frobenius": FrobeniusScalarizer(),
}


def get_scalarizer(name: str) -> BaseScalarizer:
    s = _SCALARIZERS.get(name)
    if s is None:
        raise ValueError(f"Unknown scalarizer '{name}'. Choose: {list(_SCALARIZERS)}")
    return s
