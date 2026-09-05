"""The observation the exploration agent consumes.

A deliberately narrow subset of home-robot's `Observations`: coverage
exploration needs pose, depth and (for recording only) RGB. No semantics, no
instances, no proprioception.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._vendor.interfaces import (  # noqa: F401  (re-exported for callers)
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)


@dataclass
class Observations:
    """One frame of sensor data.

    Attributes:
        gps: (2,) (x, y) in metres; +x forward, +y to the agent's left.
        compass: (1,) heading in radians; +theta is a left turn.
        rgb: (H, W, 3) uint8-valued.
        depth: (H, W) in metres.
        camera_pose: optional (4, 4) camera-to-world transform. When None the
            map module assumes a level camera at the configured height, which is
            correct for a Habitat agent that never looks up or down.
    """

    gps: np.ndarray
    compass: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    camera_pose: Optional[np.ndarray] = None
