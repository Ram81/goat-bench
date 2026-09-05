"""Frontier-exploration agent for coverage tours.

This is home-robot's `ObjectNavAgent` inference loop with the object-goal logic
removed. The pipeline per step is:

    depth + pose -> 2D occupancy/explored map -> frontier goal -> FMM plan
    -> discrete action

Compared with the home-robot original, the agent owns its own termination
policy. In `OVMMExplorationAgent` the goal is never found, so `DiscretePlanner`'s
"nowhere left to explore" STOP sits inside an `if found_goal:` branch that is
unreachable, and a wedged agent burns the entire step budget in place emitting
pixel-identical frames. Here the agent stops on its own terms instead -- see
`_termination_reason`.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ._vendor.discrete_planner import DiscretePlanner
from ._vendor import pose as pu
from ._vendor.semantic_map_state import Categorical2DSemanticMapState
from .config import ExplorationConfig
from .interfaces import DiscreteNavigationAction, Observations
from .module import FrontierExplorationModule


class FrontierExplorationAgent:
    """Explores an unknown scene by repeatedly driving to the nearest frontier."""

    def __init__(
        self,
        config: Optional[ExplorationConfig] = None,
        device_id: int = 0,
        max_steps: Optional[int] = None,
        stuck_patience: int = 30,
    ):
        """
        Arguments:
            config: exploration settings; defaults to `ExplorationConfig()`.
            device_id: CUDA device for the map update, or -1 for CPU.
            max_steps: step budget for one episode. Overrides `config.max_steps`.
            stuck_patience: end the episode after this many consecutive steps
                with no measurable pose change. 0 disables the check.
        """
        self.config = config or ExplorationConfig()
        self.max_steps = max_steps if max_steps is not None else self.config.max_steps
        self.stuck_patience = stuck_patience
        self.verbose = self.config.planner.verbose

        if device_id < 0 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        smap = self.config.semantic_map
        self.num_sem_categories = smap.num_sem_categories

        self.module = FrontierExplorationModule(self.config).to(self.device)
        self.module.eval()

        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=1,
            num_sem_categories=smap.num_sem_categories,
            map_resolution=smap.map_resolution,
            map_size_cm=smap.map_size_cm,
            global_downscaling=smap.global_downscaling,
        )

        planner_cfg = self.config.planner
        self.planner = DiscretePlanner(
            turn_angle=self.config.environment.turn_angle,
            collision_threshold=planner_cfg.collision_threshold,
            step_size=planner_cfg.step_size,
            obs_dilation_selem_radius=planner_cfg.obs_dilation_selem_radius,
            goal_dilation_selem_radius=planner_cfg.goal_dilation_selem_radius,
            map_size_cm=smap.map_size_cm,
            map_resolution=smap.map_resolution,
            visualize=self.config.visualize,
            print_images=self.config.print_images,
            dump_location=self.config.dump_location,
            exp_name=self.config.exp_name,
            agent_cell_radius=self.config.agent_cell_radius,
            min_obs_dilation_selem_radius=planner_cfg.min_obs_dilation_selem_radius,
            map_downsample_factor=planner_cfg.map_downsample_factor,
            map_update_frequency=planner_cfg.map_update_frequency,
            discrete_actions=planner_cfg.discrete_actions,
            min_goal_distance_cm=planner_cfg.min_goal_distance_cm,
            continuous_angle_tolerance=planner_cfg.continuous_angle_tolerance,
        )

        self.one_hot_encoding = torch.eye(
            smap.num_sem_categories, device=self.device
        )
        self.goal_update_steps = self.module.goal_update_steps

        self.timestep = 0
        self.timesteps_before_goal_update = 0
        self.last_pose = np.zeros(3)
        self.stuck_steps = 0
        self.finished = False

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the map and all per-episode counters."""
        self.timestep = 0
        self.timesteps_before_goal_update = 0
        self.last_pose = np.zeros(3)
        self.stuck_steps = 0
        self.finished = False
        self.semantic_map.init_map_and_pose()
        self.planner.reset()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Map the current frame, pick a frontier, and return one action."""
        obs_preprocessed, pose_delta, camera_pose = self._preprocess_obs(obs)
        planner_inputs, map_info = self._update_map_and_pick_goal(
            obs_preprocessed, pose_delta, camera_pose
        )

        reason = self._termination_reason(planner_inputs)
        if reason is not None:
            self.finished = True
            action = DiscreteNavigationAction.STOP
        elif self.timestep < self.config.panorama_start_steps:
            # Spin in place first so the map starts with a full 360-degree view;
            # otherwise the first frontier is chosen from a single frustum.
            action = DiscreteNavigationAction.TURN_RIGHT
        else:
            action, closest_goal_map, _, _ = self.planner.plan(
                **planner_inputs,
                use_dilation_for_stg=self.config.planner.use_dilation_for_stg,
                timestep=self.timestep,
                debug=self.verbose,
            )
            map_info["closest_goal_map"] = closest_goal_map

        self.timestep += 1
        info = {**planner_inputs, **map_info, "termination_reason": reason}
        return action, info

    def _termination_reason(self, planner_inputs: Dict[str, Any]) -> Optional[str]:
        """Why the episode should end now, or None to keep going."""
        if self.timestep >= self.max_steps:
            return "max_steps"
        # Unlike the goal-driven case, an empty frontier map is meaningful here:
        # every reachable region has been seen, so the tour is complete.
        if self.timestep > self.config.panorama_start_steps and not planner_inputs[
            "frontier_map"
        ].any():
            return "fully_explored"
        if self.stuck_patience and self.stuck_steps >= self.stuck_patience:
            return "stuck"
        return None

    def note_pose(self, obs: Observations) -> None:
        """Feed back the post-action pose so the agent can detect being wedged.

        Called by the collector after `apply_action`. Kept separate from `act`
        because the agent sees each observation once, and staleness has to be
        measured across the action boundary.
        """
        pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]], dtype=np.float64)
        moved = (
            np.linalg.norm(pose[:2] - self.last_pose[:2]) > 1e-3
            or abs(pose[2] - self.last_pose[2]) > 1e-3
        )
        self.stuck_steps = 0 if moved else self.stuck_steps + 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _preprocess_obs(
        self, obs: Observations
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Pack one frame into the (1, 4 + num_sem, H, W) tensor the map wants."""
        rgb = torch.from_numpy(np.ascontiguousarray(obs.rgb)).to(self.device)
        # The map module works in centimetres.
        depth = (
            torch.from_numpy(np.ascontiguousarray(obs.depth)).unsqueeze(-1).to(self.device)
            * 100.0
        )

        # A single all-zero semantic class. The channel is never read back; it
        # exists because the vendored map module always projects one.
        semantic = torch.zeros(
            obs.depth.shape[0], obs.depth.shape[1], dtype=torch.long, device=self.device
        )
        semantic = self.one_hot_encoding[semantic]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]], dtype=np.float64)
        # home-robot builds these on the CPU and relies on DataParallel's scatter
        # to move them; this agent calls the module directly, so the transfer has
        # to be explicit or the map update fails on a device mismatch.
        pose_delta = (
            torch.tensor(pu.get_rel_pose_change(curr_pose, self.last_pose))
            .unsqueeze(0)
            .to(self.device)
        )
        self.last_pose = curr_pose

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = (
                torch.tensor(np.asarray(camera_pose)).unsqueeze(0).to(self.device)
            )

        return obs_preprocessed, pose_delta, camera_pose

    def _update_map_and_pick_goal(
        self,
        obs_preprocessed: torch.Tensor,
        pose_delta: torch.Tensor,
        camera_pose: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run one map update and return the planner's inputs."""
        dones = torch.tensor([False], device=self.device)
        update_global = torch.tensor(
            [self.timesteps_before_goal_update == 0], device=self.device
        )

        (
            goal_map,
            found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs_preprocessed.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1) if camera_pose is not None else None,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()

        self.semantic_map.update_frontier_map(0, frontier_map[0][0].cpu().numpy())
        if self.timesteps_before_goal_update == 0:
            self.semantic_map.update_global_goal_for_env(0, goal_map[0])
            self.timesteps_before_goal_update = self.goal_update_steps
        self.timesteps_before_goal_update -= 1

        planner_inputs = {
            "obstacle_map": self.semantic_map.get_obstacle_map(0),
            "goal_map": self.semantic_map.get_goal_map(0),
            "frontier_map": self.semantic_map.get_frontier_map(0),
            "sensor_pose": self.semantic_map.get_planner_pose_inputs(0),
            "found_goal": bool(found_goal[0].item()),
        }
        map_info = {
            "explored_map": self.semantic_map.get_explored_map(0),
            "been_close_map": self.semantic_map.get_been_close_map(0),
            "timestep": self.timestep,
        }
        return planner_inputs, map_info
