"""Model registry: maps model name → factory function that accepts config kwargs."""
from __future__ import annotations
from typing import Callable
from .gaussian import GaussianModel
from .garch import GARCHModel
from .hawkes import HawkesModel
from .base import BaseLocalModel
from ..config import FIBConfig


def _make_gaussian(cfg: FIBConfig) -> GaussianModel:
    return GaussianModel(var_floor=cfg.var_floor)


def _make_garch(cfg: FIBConfig) -> GARCHModel:
    return GARCHModel(
        persistence_max=cfg.garch_persistence_max,
        var_floor=cfg.var_floor,
    )


def _make_hawkes(cfg: FIBConfig) -> HawkesModel:
    return HawkesModel(intensity_floor=cfg.hawkes_intensity_floor)


_FACTORIES: dict[str, Callable[[FIBConfig], BaseLocalModel]] = {
    "gaussian": _make_gaussian,
    "garch": _make_garch,
    "hawkes": _make_hawkes,
}


def create_model(cfg: FIBConfig) -> BaseLocalModel:
    """Instantiate the correct model with config-derived parameters."""
    factory = _FACTORIES.get(cfg.model)
    if factory is None:
        raise ValueError(f"Unknown model '{cfg.model}'. Available: {list(_FACTORIES)}")
    model = factory(cfg)
    model.initialize()
    return model


REGISTRY = {
    "gaussian": GaussianModel,
    "garch": GARCHModel,
    "hawkes": HawkesModel,
}

__all__ = ["create_model", "REGISTRY", "GaussianModel", "GARCHModel", "HawkesModel"]
