"""Configuration for frontier-exploration tour collection.

A plain dataclass rather than yacs/OmegaConf: the exploration agent needs ~25
knobs and pulling in home-robot's two-layer config system to carry them would be
the larger part of the port.

Defaults are copied from the home-robot settings that produced the validated
HSSD tours -- `projects/habitat_ovmm/configs/agent/heuristic_agent.yaml` and
`configs/env/hssd_explore.yaml` -- so behaviour matches that reference. Sensor
geometry defaults instead to GOAT-Bench's Stretch agent
(`config/tasks/goat_stretch_hm3d.yaml`), and `collect_tour.py` overrides
frame_height/frame_width from the live simulator config regardless.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemanticMapConfig:
    """2D map geometry and the depth-projection thresholds.

    `num_sem_categories` stays at 1: coverage exploration reads only the
    obstacle and explored channels, but the vendored map module is kept
    unmodified, and it always projects at least one semantic channel.
    """

    num_sem_categories: int = 1
    map_size_cm: int = 4800  # global map size (cm)
    map_resolution: int = 5  # size of map bins (cm)
    vision_range: int = 100  # diameter of locally visible region (cells)
    global_downscaling: int = 2  # ratio of global over local map
    du_scale: int = 4  # frame downscaling before point-cloud projection
    cat_pred_threshold: float = 1.0  # depth points per bin to assign a category
    exp_pred_threshold: float = 1.0  # depth points per bin to call it explored
    map_pred_threshold: float = 1.0  # depth points per bin to call it obstacle
    been_close_to_radius: int = 100  # radius (cm) of the been-close-to region
    explored_radius: int = 50  # radius (cm) of the visually explored region
    must_explore_close: bool = False
    min_obs_height_cm: int = 10  # minimum obstacle height (cm)
    # Depth range used for *mapping*, which is deliberately shorter than the
    # sensor's range: far returns are noisy and, projected into the map, smear
    # obstacles across cells the agent cannot actually see into. home-robot never
    # sets these and so inherits the map module's 0.5/3.5 defaults; they are
    # spelled out here because the value is load-bearing and easy to miss.
    map_min_depth: float = 0.5  # (m)
    map_max_depth: float = 3.5  # (m)
    # Median filtering to suppress spurious single-cell obstacles.
    dilate_obstacles: bool = True
    dilate_size: int = 3
    dilate_iter: int = 1
    exploration_type: str = "default"


@dataclass
class PlannerConfig:
    """FMM planner and short-term-goal settings."""

    collision_threshold: float = 0.20  # forward distance below which we collided (m)
    obs_dilation_selem_radius: int = 3  # obstacle dilation radius (cells)
    goal_dilation_selem_radius: int = 10  # goal dilation radius (cells)
    min_obs_dilation_selem_radius: int = 1
    step_size: int = 5  # max distance of the selected short-term goal
    use_dilation_for_stg: bool = False
    map_downsample_factor: float = 1.0
    map_update_frequency: int = 1
    discrete_actions: bool = True
    verbose: bool = False
    min_goal_distance_cm: float = 50.0
    continuous_angle_tolerance: float = 30.0
    explored_area_dilation_radius: int = 10


@dataclass
class EnvironmentConfig:
    """Sensor geometry. Defaults match GOAT-Bench's Stretch agent."""

    frame_height: int = 640
    frame_width: int = 360
    camera_height: float = 1.31  # camera sensor height (m)
    hfov: float = 42.0  # horizontal field of view (deg)
    turn_angle: float = 30.0  # agent turn angle (deg)
    forward: float = 0.25  # forward motion (m)
    min_depth: float = 0.0
    max_depth: float = 10.0


@dataclass
class ExplorationConfig:
    """Top-level config for the frontier exploration agent."""

    # Planner footprint radius (m). Deliberately smaller than the simulated
    # Stretch's 0.17 m collision radius -- this is the planning inflation, and
    # 0.05 is the value the reference HSSD tours were collected with. Raising it
    # to the true body radius makes the planner refuse narrow doorways.
    radius: float = 0.05
    max_steps: int = 10000  # hard cap; collect_tour.py sets the real budget
    panorama_start: bool = True  # spin 360 degrees on episode start
    exploration_strategy: str = "seen_frontier"  # or "been_close_to_frontier"

    # Debug visualisation. Off by default -- view_tour.py renders from the saved
    # frames instead, without needing the simulator.
    visualize: bool = False
    print_images: bool = False
    dump_location: str = "datadump"
    exp_name: str = "explore_goat"

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    semantic_map: SemanticMapConfig = field(default_factory=SemanticMapConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    @property
    def agent_cell_radius(self) -> int:
        """Planner footprint in map cells."""
        return int(np.ceil(self.radius * 100.0 / self.semantic_map.map_resolution))

    @property
    def panorama_start_steps(self) -> int:
        """Number of turn actions that make one full revolution."""
        if not self.panorama_start:
            return 0
        return int(360 / self.environment.turn_angle)
