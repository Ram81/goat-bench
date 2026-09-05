# Minimal stand-ins for the home_robot core interfaces the vendored planner and
# map modules expect. Kept deliberately small: coverage exploration only ever
# emits STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT.
from enum import Enum
from typing import Any

import numpy as np


class Action:
    """Base class for actions, matching home_robot.core.interfaces.Action."""


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls.

    Numeric values match home-robot's enum so vendored code that compares
    against them keeps working; the manipulation entries are unreachable here
    but retained to preserve those values.
    """

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    POST_NAV_MODE = 8
    EXTEND_ARM = 9
    EMPTY_ACTION = 10
    SNAP_OBJECT = 11
    DESNAP_OBJECT = 12
    OPEN_GRIPPER = 13
    CLOSE_GRIPPER = 14


class ContinuousNavigationAction(Action):
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError(
                "continuous navigation action space has 3 dimensions, x y and theta"
            )
        self.xyt = xyt


Pose = Any  # only used in a type hint inside the vendored map state
