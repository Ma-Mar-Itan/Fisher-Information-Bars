"""Numerical utilities for information matrix operations."""
from __future__ import annotations
import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T) / 2, enforcing exact symmetry."""
    return (A + A.T) * 0.5


def ridge_logdet(J: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute log-det of (J + eps*I) with full numerical guardrails.

    Handles:
    - Zero matrices
    - Non-PSD matrices (negative eigenvalues from numerical drift)
    - Near-singular matrices

    Returns log(det(J + eps*I)) where all eigenvalues are clipped to [eps/10, inf].
    """
    n = J.shape[0]
    M = symmetrize(J) + eps * np.eye(n)
    # Use eigvalsh (symmetric path) — faster and more stable than slogdet
    eigvals = np.linalg.eigvalsh(M)
    # Clip any numerically negative eigenvalues
    floor = max(eps * 1e-3, 1e-300)
    eigvals = np.maximum(eigvals, floor)
    return float(np.sum(np.log(eigvals)))


def safe_outer(score: np.ndarray) -> np.ndarray:
    """Outer product of score vector, with NaN/Inf guard."""
    if not np.all(np.isfinite(score)):
        n = score.shape[0]
        return np.zeros((n, n))
    return np.outer(score, score)
