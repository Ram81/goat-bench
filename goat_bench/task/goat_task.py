import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
import habitat_sim
import numpy as np
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class GoatEpisode(NavigationEpisode):
    r"""Goat Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    tasks: List[NavigationEpisode] = []

    @property
    def goals_keys(self) -> Dict:
        r"""Dictionary of goals types and corresonding keys"""
        goals_keys = {ep["task_type"]: [] for ep in self.tasks}

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = (
                    f"{os.path.basename(self.scene_id)}_{ep['object_category']}"
                )

            elif ep["task_type"] in ["imagenav", "languagenav"]:
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            goals_keys[ep["task_type"]].append(goal_key)

        return goals_keys

    def goals_keys_with_sequence(self) -> str:
        r"""The key to retrieve the goals"""
        goals_keys = []

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = (
                    f"{os.path.basename(self.scene_id)}_{ep['object_category']}"
                )

            elif ep["task_type"] in ["imagenav", "languagenav"]:
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            # elif ep["task_type"] == "languagenav":
            #     goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_instance_id']}"

            goals_keys.append(goal_key)

        return goals_keys


@registry.register_task(name="Goat-v1")
class GoatTask(NavigationTask):  # TODO
    r"""A GOAT Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    is_sub_task_stop_called: bool = False
    active_subtask_idx: int = 0
    last_action: Optional[SimulatorTaskAction] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sub_task_stop_called = False
        self.active_subtask_idx = 0
        self.last_action = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.is_sub_task_stop_called = False
        self.active_subtask_idx = 0
        self.last_action = None
        return super().reset(*args, **kwargs)

    def _subtask_stop_called(self, *args: Any, **kwargs: Any) -> bool:
        return isinstance(self.last_action, SubtaskStopAction)

    def _check_episode_is_active(
        self, episode, *args: Any, **kwargs: Any
    ) -> bool:
        return not getattr(
            self, "is_stop_called", False
        ) and self.active_subtask_idx < len(episode.goals)

    def step(self, action: Dict[str, Any], episode: GoatEpisode):
        action_name = action["action"]
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        task_action = self.actions[action_name]
        observations = super().step(action, episode)
        self.last_action = task_action
        return observations


@registry.register_task_action
class SubtaskStopAction(SimulatorTaskAction):
    name: str = "subtask_stop"

    def reset(self, task: GoatTask, *args: Any, **kwargs: Any):
        task.is_sub_task_stop_called = False  # type: ignore
        task.active_subtask_idx = 0

    def step(self, task: GoatTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_sub_task_stop_called = True  # type: ignore
        task.active_subtask_idx += 1
        return self._sim.get_observations_at()  # type: ignore
