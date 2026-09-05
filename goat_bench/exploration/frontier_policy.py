"""Frontier selection for coverage exploration.

Derived from home-robot's `ObjectNavFrontierExplorationPolicy`, with the
goal-object branches removed: there is no goal here, so the policy always
returns the frontier and `found_goal` is always False. That difference matters
downstream -- see the note in `agent.py` about STOP being unreachable.
"""
import skimage.morphology
import torch
import torch.nn as nn

from ._vendor.constants import MapConstants as MC
from ._vendor.morphology import binary_dilation


class FrontierExplorationPolicy(nn.Module):
    """Select the border between explored and unexplored space as the goal.

    Stateless and parameter-free; it is an `nn.Module` only so the structuring
    elements move to the right device with the rest of the model.
    """

    def __init__(
        self,
        exploration_strategy: str = "seen_frontier",
        explored_area_dilation_radius: int = 10,
    ):
        super().__init__()
        if exploration_strategy not in ("seen_frontier", "been_close_to_frontier"):
            raise ValueError(
                f"unknown exploration_strategy {exploration_strategy!r}; expected "
                "'seen_frontier' or 'been_close_to_frontier'"
            )
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(explored_area_dilation_radius))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    def goal_update_steps(self) -> int:
        return 1

    def get_frontier_map(self, map_features: torch.Tensor) -> torch.Tensor:
        """Frontier cells of shape (batch_size, 1, M, M).

        Arguments:
            map_features: (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
        """
        # Unexplored area.
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        else:
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()

        # Erode the unexplored region (equivalently, dilate the explored one) so
        # the frontier sits clear of the explored boundary rather than hugging
        # it, which otherwise makes the short-term goal unreachable.
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # One-cell border of what remains.
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map

    def forward(self, map_features: torch.Tensor):
        """Returns (goal_map, found_goal).

        `found_goal` is always zero -- there is nothing to find while mapping for
        coverage -- and is returned only to keep the planner's interface intact.
        """
        frontier_map = self.get_frontier_map(map_features)
        goal_map = frontier_map.squeeze(1)
        found_goal = torch.zeros(
            map_features.shape[0], dtype=torch.bool, device=map_features.device
        )
        return goal_map, found_goal
