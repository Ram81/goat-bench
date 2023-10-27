from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, TopDownMap
from habitat.utils.visualizations import fog_of_war, maps

from goat.task.goat_task import SubtaskStopAction
from goat.utils.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class OVONDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.distance_to == "VIEW_POINTS":
            goals = task._dataset.goals_by_category[episode.goals_key]
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in goals
                for view_point in goal.view_points
            ]

            if episode.children_object_categories is not None:
                for children_category in episode.children_object_categories:
                    scene_id = episode.scene_id.split("/")[-1]
                    goal_key = f"{scene_id}_{children_category}"

                    # Ignore if there are no valid viewpoints for goal
                    if goal_key not in task._dataset.goals_by_category:
                        continue
                    self._episode_view_points.extend(
                        [
                            vp.agent_state.position
                            for goal in task._dataset.goals_by_category[
                                goal_key
                            ]
                            for vp in goal.view_points
                        ]
                    )

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "POINT":
                goals = task._dataset.goals_by_category[episode.goals_key]
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in goals],
                    episode,
                )
            elif self._config.distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class OVONObjectGoalID(Measure):
    cls_uuid: str = "object_goal_id"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        cache = load_pickle(config.cache)
        self.vocab = sorted(list(cache.keys()))
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = self.vocab.index(episode.object_category)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        pass


@registry.register_measure
class GoatDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._current_subtask_idx = 0
        self.prev_distance_to_target = 0

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = {"distance_to_target": 0, "prev_distance_to_target": 0}
        self._current_subtask_idx = 0
        self.prev_distance_to_target = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        self.prev_distance_to_target = self._metric["distance_to_target"]
        if self._current_subtask_idx != task.active_subtask_idx:
            self._previous_position = None
            self._current_subtask_idx = task.active_subtask_idx
            episode._shortest_path_cache = None

        if self._current_subtask_idx == len(episode.tasks):
            self._metric = {
                "distance_to_target": self._metric["distance_to_target"],
                "prev_distance_to_target": self.prev_distance_to_target,
            }
            return

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "VIEW_POINTS":
                viewpoints = [
                    view_point["agent_state"]["position"]
                    for goal in episode.goals[task.active_subtask_idx]
                    for view_point in goal["view_points"]
                ]
                distance_to_target = self._sim.geodesic_distance(
                    current_position, viewpoints, None
                )
            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )
                raise NotImplementedError

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = {
                "distance_to_target": distance_to_target,
                "prev_distance_to_target": self.prev_distance_to_target,
            }
        if not np.isfinite(
            self._metric["distance_to_target"]
        ) or not np.isfinite(self._metric["prev_distance_to_target"]):
            print(
                current_position,
                self._previous_position,
                task.last_action,
                episode.tasks[task.active_subtask_idx],
                self._metric,
            )


@registry.register_measure
class GoatSuccess(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "success"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._current_subtask_idx = 0
        self._success_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid]
        )
        self._previous_position = None
        self._metric = None
        self._current_subtask_idx = 0
        self._success_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        self._subtask_success = [0.0] * len(episode.tasks)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        if self._current_subtask_idx == len(episode.tasks):
            return

        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()

        if (
            task._subtask_stop_called()
            and distance_to_target["prev_distance_to_target"]
            < self._config.success_distance
        ):
            self._success_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += 1
            self._subtask_success[self._current_subtask_idx] = 1.0

        success_by_subtask = {}
        for k in ["object", "image", "description"]:
            if self._success_by_subtasks[k] == 0:
                success_by_subtask["{}_success".format(k)] = 0.0
            else:
                success_by_subtask["{}_success".format(k)] = (
                    self._success_by_subtasks[k] / self._subtask_counts[k]
                )

        num_subtask_success = sum(self._subtask_success) == len(episode.tasks)
        self._metric = {
            "task_success": num_subtask_success
            and getattr(task, "is_stop_called", False),
            "composite_success": num_subtask_success,
            "partial_success": sum(self._success_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            # **success_by_subtask,
            "subtask_success": self._subtask_success,
            **success_by_subtask,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx


@registry.register_measure
class GoatSPL(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "spl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._current_subtask_idx = 0
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid, GoatSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self._current_subtask_idx = 0

        self._spl_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1
            self._spl_by_subtasks["{}_spl".format(t[1])] = 0

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[
            GoatSuccess.cls_uuid
        ].get_metric()["subtask_success"]

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if task._subtask_stop_called():
            spl = ep_success[self._current_subtask_idx] * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance,
                    self._agent_episode_distance,
                )
            )
            self._spl_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += spl

        spl_by_subtask = {}
        for k, v in self._subtask_counts.items():
            spl_by_subtask["{}_spl".format(k)] = (
                self._spl_by_subtasks[k] / self._subtask_counts[k]
            )

        self._metric = {
            "composite_spl": sum(self._spl_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            # **spl_by_subtask,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx
            self._start_end_episode_distance = task.measurements.measures[
                GoatDistanceToGoal.cls_uuid
            ].get_metric()["distance_to_target"]
            self._agent_episode_distance = 0


@registry.register_measure
class GoatSoftSPL(Measure):
    """The measure calculates a SoftSPL."""

    cls_uuid: str = "soft_spl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._current_subtask_idx = 0
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "soft_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid, GoatSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self._current_subtask_idx = 0

        self._softspl_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1
            self._softspl_by_subtasks["{}_softspl".format(t[1])] = 0

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["prev_distance_to_target"]

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if task._subtask_stop_called():
            ep_soft_success = max(
                0, (1 - distance_to_target / self._start_end_episode_distance)
            )
            soft_spl = ep_soft_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance,
                    self._agent_episode_distance,
                )
            )
            self._softspl_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += soft_spl

        softspl_by_subtask = {}
        for k, v in self._subtask_counts.items():
            softspl_by_subtask["{}_softspl".format(k)] = (
                self._softspl_by_subtasks[k] / self._subtask_counts[k]
            )

        self._metric = {
            "composite_softspl": sum(self._softspl_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            # **softspl_by_subtask,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx
            self._start_end_episode_distance = task.measurements.measures[
                GoatDistanceToGoal.cls_uuid
            ].get_metric()["distance_to_target"]
            self._agent_episode_distance = 0


@registry.register_measure
class GoatDistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "goat_distance_to_goal_reward"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()
        # print(distance_to_target, self._previous_distance)

        subtask_success_reward = 0
        # Handle case when subtask stop is called
        if task._subtask_stop_called():
            self._previous_distance = distance_to_target["distance_to_target"]

            if (
                distance_to_target["prev_distance_to_target"]
                < self._config.success_distance
            ):
                subtask_success_reward = 5.0

        self._metric = (
            -(
                distance_to_target["distance_to_target"]
                - self._previous_distance
            )
            + subtask_success_reward
        )
        self._previous_distance = distance_to_target["distance_to_target"]


@registry.register_measure
class GoatTopDownMap(TopDownMap):
    r"""Top Down Map measure for GOAT task."""

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "goat_top_down_map"

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for super_goals in episode.goals:
                if type(super_goals[0]) != dict:
                    super_goals = super_goals[0]
                for goal in super_goals:
                    if self._is_on_same_floor(goal["position"][1]):
                        try:
                            if goal["view_points"] is not None:
                                for view_point in goal["view_points"]:
                                    self._draw_point(
                                        view_point["agent_state"]["position"],
                                        maps.MAP_VIEW_POINT_INDICATOR,
                                    )
                        except AttributeError:
                            pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:
            for super_goals in episode.goals:
                if type(super_goals[0]) != dict:
                    super_goals = super_goals[0]
                for goal in super_goals:
                    if self._is_on_same_floor(goal["position"][1]):
                        try:
                            self._draw_point(
                                goal["position"],
                                maps.MAP_TARGET_POINT_INDICATOR,
                            )
                        except AttributeError:
                            pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for super_goals in episode.goals:
                for goal in super_goals:
                    try:
                        sem_scene = self._sim.semantic_annotations()
                        object_id = goal.object_id
                        if type(object_id) != int:
                            object_id = int(object_id.split("_")[-1])
                        assert int(
                            sem_scene.objects[object_id].id.split("_")[-1]
                        ) == int(
                            object_id
                        ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                        center = sem_scene.objects[object_id].aabb.center
                        x_len, _, z_len = (
                            sem_scene.objects[object_id].aabb.sizes / 2.0
                        )
                        # Nodes to draw rectangle
                        corners = [
                            center + np.array([x, 0, z])
                            for x, z in [
                                (-x_len, -z_len),
                                (-x_len, z_len),
                                (x_len, z_len),
                                (x_len, -z_len),
                                (-x_len, -z_len),
                            ]
                            if self._is_on_same_floor(center[1])
                        ]

                        map_corners = [
                            maps.to_grid(
                                p[2],
                                p[0],
                                (
                                    self._top_down_map.shape[0],
                                    self._top_down_map.shape[1],
                                ),
                                sim=self._sim,
                            )
                            for p in corners
                        ]

                        maps.draw_path(
                            self._top_down_map,
                            map_corners,
                            maps.MAP_TARGET_BOUNDING_BOX,
                            self.line_thickness,
                        )
                    except AttributeError:
                        pass

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            # self._draw_shortest_path(episode, agent_position)
        if self._config.draw_source:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )
