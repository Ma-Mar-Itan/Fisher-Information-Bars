"""Abstract base class for local likelihood models."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..events import MarketEvent


class BaseLocalModel(ABC):
    """
    Contract for all local likelihood models used by FIBBuilder.

    Each model maintains its own parameter state and exposes:
    - n_params          : dimensionality of theta
    - initialize()      : reset to defaults
    - update(event)     : ingest one event, update theta
    - score(event)      : gradient of log-lik at current theta (before update)
    - observed_information_increment(event)  : OPG = score @ score.T
    - expected_information_increment(event)  : analytic Fisher info (if available)
    - state_dict()      : serializable snapshot of current state
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of model parameters."""
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Reset model to initial state."""
        ...

    @abstractmethod
    def update(self, event: MarketEvent) -> None:
        """Ingest one event and update model state."""
        ...

    @abstractmethod
    def score(self, event: MarketEvent) -> np.ndarray:
        """
        Return score vector (gradient of log-lik) at current theta
        evaluated at the new event, BEFORE updating state.
        Shape: (n_params,)
        """
        ...

    def observed_information_increment(self, event: MarketEvent) -> np.ndarray:
        """OPG approximation: score @ score.T. Shape: (n_params, n_params)."""
        s = self.score(event)
        if not np.all(np.isfinite(s)):
            return np.zeros((self.n_params, self.n_params))
        return np.outer(s, s)

    def expected_information_increment(self, event: MarketEvent) -> np.ndarray:
        """
        Analytic Fisher information increment.
        Default: falls back to OPG. Override in subclasses for closed-form.
        """
        return self.observed_information_increment(event)

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable state snapshot."""
        ...
