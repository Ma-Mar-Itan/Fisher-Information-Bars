"""Abstract base for local likelihood models."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from ..events import MarketEvent
from ..types import FloatArray, Matrix


class BaseLocalModel(ABC):
    """Contract every model must satisfy."""

    name: str = "base"

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def update(self, event: MarketEvent) -> None:
        """Ingest event and update internal parameter state."""

    @abstractmethod
    def score(self, event: MarketEvent) -> FloatArray:
        """Return score vector s_i at current theta."""

    def observed_information_increment(self, event: MarketEvent) -> Matrix:
        """OPG: s_i s_i^T."""
        s = self.score(event)
        return np.outer(s, s)

    def expected_information_increment(self, event: MarketEvent) -> Matrix:
        """Default: same as observed (OPG); override for analytic forms."""
        return self.observed_information_increment(event)

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...
