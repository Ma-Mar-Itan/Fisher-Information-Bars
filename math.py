"""Numerical helpers."""
from __future__ import annotations
import numpy as np
from ..types import Matrix


def symmetrize(J: Matrix) -> Matrix:
    """Return (J + J^T) / 2."""
    return (J + J.T) * 0.5


def ridge_logdet(J: Matrix, eps: float = 1e-6) -> float:
    """log det(J + eps*I), numerically stable via slogdet."""
    n = J.shape[0]
    M = symmetrize(J) + eps * np.eye(n)
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        # Fall back to eigenvalue floor
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.maximum(eigvals, eps)
        return float(np.sum(np.log(eigvals)))
    return float(logdet)


def safe_inv(J: Matrix, eps: float = 1e-6) -> Matrix:
    n = J.shape[0]
    M = symmetrize(J) + eps * np.eye(n)
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.eye(n) / eps
