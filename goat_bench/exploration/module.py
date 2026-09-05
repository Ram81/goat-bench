"""Map update + frontier selection, bundled as one nn.Module.

Mirrors home-robot's `ObjectNavAgentModule` minus the goal-category plumbing.
Kept as a module (rather than folded into the agent) so the whole map update can
be moved to a GPU in one `.to(device)`.
"""
from typing import Tuple

import torch
import torch.nn as nn

from ._vendor.semantic_map_module import Categorical2DSemanticMapModule
from .config import ExplorationConfig
from .frontier_policy import FrontierExplorationPolicy


class FrontierExplorationModule(nn.Module):
    def __init__(self, config: ExplorationConfig):
        super().__init__()
        env, smap = config.environment, config.semantic_map
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=env.frame_height,
            frame_width=env.frame_width,
            camera_height=env.camera_height,
            hfov=env.hfov,
            num_sem_categories=smap.num_sem_categories,
            map_size_cm=smap.map_size_cm,
            map_resolution=smap.map_resolution,
            vision_range=smap.vision_range,
            explored_radius=smap.explored_radius,
            been_close_to_radius=smap.been_close_to_radius,
            global_downscaling=smap.global_downscaling,
            du_scale=smap.du_scale,
            cat_pred_threshold=smap.cat_pred_threshold,
            exp_pred_threshold=smap.exp_pred_threshold,
            map_pred_threshold=smap.map_pred_threshold,
            min_depth=smap.map_min_depth,
            max_depth=smap.map_max_depth,
            must_explore_close=smap.must_explore_close,
            min_obs_height_cm=smap.min_obs_height_cm,
            dilate_obstacles=smap.dilate_obstacles,
            dilate_size=smap.dilate_size,
            dilate_iter=smap.dilate_iter,
            record_instance_ids=False,
            instance_memory=None,
            max_instances=0,
            evaluate_instance_tracking=False,
            exploration_type=smap.exploration_type,
        )
        self.policy = FrontierExplorationPolicy(
            exploration_strategy=config.exploration_strategy,
            explored_area_dilation_radius=config.planner.explored_area_dilation_radius,
        )

    @property
    def goal_update_steps(self) -> int:
        return self.policy.goal_update_steps

    def forward(
        self,
        seq_obs: torch.Tensor,
        seq_pose_delta: torch.Tensor,
        seq_dones: torch.Tensor,
        seq_update_global: torch.Tensor,
        seq_camera_poses: torch.Tensor,
        init_local_map: torch.Tensor,
        init_global_map: torch.Tensor,
        init_local_pose: torch.Tensor,
        init_global_pose: torch.Tensor,
        init_lmb: torch.Tensor,
        init_origins: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Update the map over a sequence of frames and pick frontier goals.

        Arguments:
            seq_obs: (batch, seq_len, 3 + 1 + num_sem_categories, H, W)
            seq_pose_delta: (batch, seq_len, 3) pose change since the last frame
            seq_dones: (batch, seq_len) episode-restart flags
            seq_update_global: (batch, seq_len) whether to sync the global map
            seq_camera_poses: (batch, seq_len, 4, 4)

        Returns:
            seq_goal_map, seq_found_goal, seq_frontier_map, final_local_map,
            final_global_map, seq_local_pose, seq_global_pose, seq_lmb,
            seq_origins
        """
        batch_size, sequence_length = seq_obs.shape[:2]

        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        map_features = seq_map_features.flatten(0, 1)
        goal_map, found_goal = self.policy(map_features)
        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])
        seq_found_goal = found_goal.view(batch_size, sequence_length)

        frontier_map = self.policy.get_frontier_map(map_features)
        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
