"""Frontier-based exploration for coverage tours in GOAT-Bench scenes.

Ported from home-robot's OVMM exploration stack (`projects/habitat_ovmm`), with
the object-goal, semantic-perception and manipulation machinery removed: the
only task here is to cover a scene and record what the agent saw.

The heavier numerical pieces -- depth projection into a 2D map, the FMM planner,
the discrete motion planner -- are vendored verbatim under `_vendor/` so their
tuned behaviour is preserved exactly; see `_vendor/README.md`.
"""
from .agent import FrontierExplorationAgent
from .config import (
    EnvironmentConfig,
    ExplorationConfig,
    PlannerConfig,
    SemanticMapConfig,
)
from .habitat_env import HabitatExplorationEnv
from .interfaces import DiscreteNavigationAction, Observations

__all__ = [
    "DiscreteNavigationAction",
    "EnvironmentConfig",
    "ExplorationConfig",
    "FrontierExplorationAgent",
    "HabitatExplorationEnv",
    "Observations",
    "PlannerConfig",
    "SemanticMapConfig",
]
