"""Scalarizers that map an information matrix to a real-valued knowledge metric."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import Matrix
from ..utils.math import ridge_logdet, symmetrize


class BaseScalarizer(ABC):
    name: str = "base"

    @abstractmethod
    def __call__(self, J: Matrix, eps: float = 1e-6) -> float: ...


class LogdetScalarizer(BaseScalarizer):
    """Phi(J) = log det(J + eps*I).  Reparameterization-invariant."""
    name = "logdet"

    def __call__(self, J: Matrix, eps: float = 1e-6) -> float:
        return ridge_logdet(J, eps=eps)


class TraceScalarizer(BaseScalarizer):
    """Phi(J) = tr(J).  Equivalent to sum of eigenvalues (A-optimality)."""
    name = "trace"

    def __call__(self, J: Matrix, eps: float = 1e-6) -> float:
        return float(np.trace(symmetrize(J)) + eps * J.shape[0])


class FrobeniusScalarizer(BaseScalarizer):
    """Phi(J) = ||J||_F."""
    name = "frobenius"

    def __call__(self, J: Matrix, eps: float = 1e-6) -> float:
        return float(np.linalg.norm(symmetrize(J), "fro") + eps)


def get_scalarizer(name: str) -> BaseScalarizer:
    mapping = {
        "logdet": LogdetScalarizer(),
        "trace": TraceScalarizer(),
        "frobenius": FrobeniusScalarizer(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown scalarizer '{name}'. Choose from {list(mapping)}")
    return mapping[name]
