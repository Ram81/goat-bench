"""Adapter between a Habitat env and the exploration agent.

Handles the two conversions that are easy to get subtly wrong: the GPS frame and
the depth encoding.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._vendor.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from .interfaces import DiscreteNavigationAction, Observations

# Habitat action names for the discrete navigation subset we emit.
ACTION_NAMES = {
    DiscreteNavigationAction.STOP: "stop",
    DiscreteNavigationAction.MOVE_FORWARD: "move_forward",
    DiscreteNavigationAction.TURN_LEFT: "turn_left",
    DiscreteNavigationAction.TURN_RIGHT: "turn_right",
}


class HabitatExplorationEnv:
    """Wraps a `habitat.Env` and speaks the agent's `Observations` dialect."""

    def __init__(
        self,
        env,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
        normalize_depth: bool = True,
        rgb_key: str = "rgb",
        depth_key: str = "depth",
        navmesh_settings: Optional[Dict[str, float]] = None,
    ):
        """
        Arguments:
            env: a constructed `habitat.Env`.
            min_depth/max_depth: the depth sensor's configured range (m). Used
                only when `normalize_depth` is True.
            normalize_depth: whether the sensor emits depth in [0, 1] (Habitat's
                `normalize_depth: True`) rather than raw metres.
            navmesh_settings: when given, recompute each scene's navmesh for
                this agent body: keys `agent_height`, `agent_radius`,
                `agent_max_climb`, `cell_height`. See `_recompute_navmesh`.
        """
        self.env = env
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.normalize_depth = normalize_depth
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.navmesh_settings = navmesh_settings
        self._navmesh_scene = None
        self._available_actions = set(env.task.actions.keys())

    def _recompute_navmesh(self) -> None:
        """Rebuild the current scene's navmesh for the configured agent body.

        GOAT-Bench does this in its own simulator (`GOATSim-v0`) because the
        navmesh shipped with a scene is built for whatever agent the dataset
        authors used; traversability -- which doorways and gaps count as
        passable -- depends on the agent's radius and climb height. Exploring
        "the GOAT-Bench scenes" with a different navmesh would explore a
        different free space.

        With the default config this is dead weight -- `GOATSim-v0` does the
        work itself, and `collect_tour.py` leaves `navmesh_settings` unset so
        it is not repeated. It exists for habitat-sim 0.2.5, where `GOATSim-v0`
        cannot be constructed at all: it calls `recompute_navmesh(...,
        include_static_objects=False)`, and 0.2.5 removed that keyword. The
        call below tries the 0.2.3 signature first and falls back to the 0.2.5
        one, so the wrapper works on either.
        """
        import habitat_sim

        sim = self.env.sim
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = self.navmesh_settings["agent_height"]
        settings.agent_radius = self.navmesh_settings["agent_radius"]
        settings.agent_max_climb = self.navmesh_settings["agent_max_climb"]
        settings.cell_height = self.navmesh_settings["cell_height"]
        try:
            sim.recompute_navmesh(
                sim.pathfinder, settings, include_static_objects=False
            )
        except TypeError:
            # habitat-sim >= 0.2.4 dropped include_static_objects. Static
            # objects are part of the scene mesh in HM3D/MP3D anyway, so the
            # resulting navmesh is the same.
            sim.recompute_navmesh(sim.pathfinder, settings)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def _preprocess_gps(self, gps: np.ndarray) -> np.ndarray:
        """Habitat GPS -> the agent's (forward, left) frame.

        Habitat's `EpisodicGPSSensor` reports (-z, x) in the episode's start
        frame: index 0 is forward, index 1 is to the agent's *right*. The map
        module's convention is +y to the left, so the second component flips.
        """
        return np.array([gps[0], -1.0 * gps[1]], dtype=np.float32)

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """Habitat depth -> metres, with out-of-range pixels sentinel-marked.

        Habitat clamps unreturned pixels to the ends of the range, which after
        rescaling are indistinguishable from genuine readings at exactly
        min/max depth. They are replaced with sentinels far beyond the mapping
        cutoff so the map module's `depth > max_depth` test discards them
        instead of projecting a wall at the sensor's range limit.
        """
        if depth.ndim == 3:
            depth = depth[:, :, -1]
        if not self.normalize_depth:
            return depth
        rescaled = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled

    def _preprocess_obs(self, habitat_obs: Dict[str, Any]) -> Observations:
        return Observations(
            rgb=habitat_obs[self.rgb_key],
            depth=self._preprocess_depth(habitat_obs[self.depth_key]),
            gps=self._preprocess_gps(np.asarray(habitat_obs["gps"])),
            compass=np.asarray(habitat_obs["compass"]),
            # None means "level camera at the configured height", which holds
            # for an agent restricted to move/turn actions.
            camera_pose=None,
        )

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    def reset(self) -> Observations:
        habitat_obs = self.env.reset()
        if self.navmesh_settings is not None:
            scene_id = self.env.current_episode.scene_id
            if scene_id != self._navmesh_scene:
                # Once per scene, not per episode: rebuilding is expensive and
                # the result depends only on the scene and the agent body.
                #
                # Done after reset rather than during reconfigure (where
                # GOATSim-v0 does it) because the wrapper cannot hook the
                # simulator's reconfigure. The effect is the same: the agent is
                # placed at the episode's start pose either way, and the new
                # navmesh is in force before the first action is taken. Note
                # that observations do not depend on the navmesh -- only motion
                # does, via habitat's step filter.
                self._recompute_navmesh()
                self._navmesh_scene = scene_id
        return self._preprocess_obs(habitat_obs)

    def apply_action(
        self, action: DiscreteNavigationAction
    ) -> Tuple[Observations, bool]:
        """Step the simulator; returns (observation, done)."""
        name = ACTION_NAMES.get(action)
        if name is None:
            raise ValueError(f"{action} is not a navigation action")
        if name not in self._available_actions:
            raise KeyError(
                f"action {name!r} is not in the task's action space "
                f"({sorted(self._available_actions)}); check the task config"
            )
        # Step by name, never by index. Habitat resolves an integer action
        # against `task.actions`, whose order need not match
        # `env.action_space.spaces` -- indexing through the latter silently
        # executed the wrong action (move_forward arriving as stop, ending the
        # episode on the first step).
        habitat_obs = self.env.step(name)
        return self._preprocess_obs(habitat_obs), self.env.episode_over

    def get_current_episode(self):
        return self.env.current_episode

    @property
    def number_of_episodes(self) -> Optional[int]:
        return getattr(self.env, "number_of_episodes", None)

    def close(self) -> None:
        self.env.close()
