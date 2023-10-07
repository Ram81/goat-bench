from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, TopDownMap
from habitat.utils.visualizations import fog_of_war, maps

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
class GOATTopDownMap(TopDownMap):
    r"""Top Down Map measure for GOAT task."""

    def __init__(
        self,
        sim: "HabitatSim",
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
