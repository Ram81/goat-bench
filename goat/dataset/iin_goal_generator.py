import argparse
import gzip
import itertools
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import habitat_sim
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim._ext.habitat_sim_bindings import SemanticObject
from habitat_sim.agent.agent import AgentConfiguration, AgentState, SixDOFPose
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from numpy import ndarray
from tqdm import tqdm

from goat.dataset.pointcloud_utils import (mesh_to_scene,
                                           points_to_surface_area,
                                           points_to_surface_area_in_scene,
                                           project_semantics_to_world)
from goat.dataset.pose_sampler import PoseSampler
from goat.dataset.semantic_utils import (ObjectCategoryMapping,
                                         get_hm3d_semantic_scenes)
from goat.dataset.visualization import (clear_log, log_text, plot_area,
                                        save_candidate_imgs)


class ImageGoalGenerator:
    ISLAND_RADIUS_LIMIT: float = 1.5

    semantic_spec_filepath: str
    img_size: Tuple[int, int]
    agent_height: float
    agent_radius: float
    pose_sampler_args: Dict[str, Any]
    min_object_coverage: float
    frame_cov_thresh_line: Tuple[float, float]
    voxel_size: float
    dbscan_slack: float
    goal_vp_cell_size: float
    goal_vp_max_dist: float
    start_poses_per_obj: float
    start_poses_tilt_angle: float
    start_distance_limits: Tuple[float, float]
    min_geo_to_euc_ratio: float
    start_retries: int
    single_floor_threshold: float
    keep_metadata: bool
    cat_map: ObjectCategoryMapping
    scene: Optional[str]
    metadata: Optional[Dict[str, Any]]

    def __init__(
        self,
        semantic_spec_filepath: str,
        img_size: Tuple[int, int],
        agent_height: float,
        agent_radius: float,
        pose_sampler_args: Dict[str, Any],
        category_mapping_file: str,
        categories: str,
        min_object_coverage: float,
        frame_cov_thresh_line: Tuple[float, float],
        voxel_size: float = 0.05,
        dbscan_slack: float = 0.01,
        goal_vp_cell_size: float = 0.1,
        goal_vp_max_dist: float = 1.0,
        start_poses_per_obj: int = 500,
        start_poses_tilt_angle: float = 30.0,
        start_distance_limits: Tuple[float, float] = (1.0, 30.0),
        min_geo_to_euc_ratio: float = 1.05,
        start_retries: int = 1000,
        single_floor_threshold: float = 0.25,
        keep_metadata: bool = False,
        verbose: bool = False,
    ) -> None:
        self.semantic_spec_filepath = semantic_spec_filepath
        self.img_size = img_size
        self.agent_height = agent_height
        self.agent_radius = agent_radius
        self.pose_sampler_args = pose_sampler_args
        self.min_object_coverage = min_object_coverage
        self.frame_cov_thresh_line = frame_cov_thresh_line
        self.voxel_size = voxel_size
        self.dbscan_slack = dbscan_slack
        self.goal_vp_cell_size = goal_vp_cell_size
        self.goal_vp_max_dist = goal_vp_max_dist
        self.start_poses_per_obj = start_poses_per_obj
        self.start_poses_tilt_angle = start_poses_tilt_angle
        self.start_distance_limits = start_distance_limits
        self.min_geo_to_euc_ratio = min_geo_to_euc_ratio
        self.start_retries = start_retries
        self.single_floor_threshold = single_floor_threshold
        self.keep_metadata = keep_metadata
        self.cat_map = ObjectCategoryMapping(
            mapping_file=category_mapping_file, allowed_categories=categories
        )
        self.verbose = verbose
        self.scene = None
        self.metadata = [] if self.keep_metadata else None

    def _config_sim(self, scene: str) -> Simulator:
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = self.semantic_spec_filepath
        sim_cfg.scene_id = scene

        sensor_specs = []
        for name, sensor_type in zip(
            ["color", "depth", "semantic"],
            [
                habitat_sim.SensorType.COLOR,
                habitat_sim.SensorType.DEPTH,
                habitat_sim.SensorType.SEMANTIC,
            ],
        ):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = f"{name}_sensor"
            sensor_spec.sensor_type = sensor_type
            sensor_spec.resolution = [self.img_size[0], self.img_size[1]]
            sensor_spec.position = [0.0, self.agent_height, 0.0]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration(
            height=self.agent_height,
            radius=self.agent_radius,
            sensor_specifications=sensor_specs,
            action_space={
                "look_up": habitat_sim.ActionSpec(
                    "look_up",
                    habitat_sim.ActuationSpec(
                        amount=self.start_poses_tilt_angle
                    ),
                ),
                "look_down": habitat_sim.ActionSpec(
                    "look_down",
                    habitat_sim.ActuationSpec(
                        amount=self.start_poses_tilt_angle
                    ),
                ),
            },
        )

        sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )

        # set the navmesh
        assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agent_height
        navmesh_settings.agent_radius = self.agent_radius
        navmesh_settings.agent_max_climb = 0.10
        navmesh_settings.cell_height = 0.05
        navmesh_success = sim.recompute_navmesh(
            sim.pathfinder, navmesh_settings, include_static_objects=False
        )
        assert navmesh_success, "Failed to build the navmesh!"
        return sim

    def _threshold_image_goals(
        self,
        frame_covs: List[float],
        object_covs: List[float],
        obj_surface_area: float,
    ) -> List[bool]:
        assert len(frame_covs) == len(object_covs)

        m, b = self.frame_cov_thresh_line
        obj_cov_thresh = m * obj_surface_area + b

        keep_goal = []
        for fc, oc in zip(frame_covs, object_covs):
            keep_goal.append(
                oc > self.min_object_coverage and fc > obj_cov_thresh
            )
        return keep_goal

    def _make_object_viewpoints(self, sim: Simulator, obj: SemanticObject):
        object_position = obj.aabb.center
        eps = 1e-5
        x_len, _, z_len = obj.aabb.sizes / 2.0 + self.goal_vp_max_dist
        x_bxp = (
            np.arange(-x_len, x_len + eps, step=self.goal_vp_cell_size)
            + object_position[0]
        )
        z_bxp = (
            np.arange(-z_len, z_len + eps, step=self.goal_vp_cell_size)
            + object_position[2]
        )
        candiatate_poses = [
            np.array([x, object_position[1], z])
            for x, z in itertools.product(x_bxp, z_bxp)
        ]

        def _down_is_navigable(pt):
            pf = sim.pathfinder

            delta_y = 0.05
            max_steps = int(2 / delta_y)
            step = 0
            is_navigable = pf.is_navigable(pt, 2)
            while not is_navigable:
                pt[1] -= delta_y
                is_navigable = pf.is_navigable(pt)
                step += 1
                if step == max_steps:
                    return False
            return True

        def _face_object(object_position: np.array, point: ndarray):
            EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
            cam_normal = (object_position - point) + EPS_ARRAY
            cam_normal[1] = 0
            cam_normal = cam_normal / np.linalg.norm(cam_normal)
            return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

        def _get_iou(pt):
            obb = habitat_sim.geo.OBB(obj.aabb)
            if obb.distance(pt) > self.goal_vp_max_dist:
                return -0.5, pt, None

            if not _down_is_navigable(pt):
                return -1.0, pt, None

            pt = np.array(sim.pathfinder.snap_point(pt))
            q = _face_object(object_position, pt)

            cov = 0
            sim.agents[0].set_state(AgentState(position=pt, rotation=q))
            for act in ["look_down", "look_up", "look_up"]:
                obs = sim.step(act)
                cov += self._compute_frame_coverage(obs, obj.semantic_id)

            return cov, pt, q

        candiatate_poses_ious = [_get_iou(pos) for pos in candiatate_poses]
        best_iou = (
            max(v[0] for v in candiatate_poses_ious)
            if len(candiatate_poses_ious) != 0
            else 0
        )
        if best_iou <= 0.0:
            return []

        view_locations = [
            {
                "agent_state": {
                    "position": pt.tolist(),
                    "rotation": quat_to_coeffs(q).tolist(),
                },
                "iou": iou,
            }
            for iou, pt, q in candiatate_poses_ious
            if iou > 0.0
        ]
        view_locations = sorted(
            view_locations, reverse=True, key=lambda v: v["iou"]
        )

        if self.verbose:
            plot_area(
                self.scene,
                obj,
                candiatate_poses_ious,
                [v["agent_state"]["position"] for v in view_locations],
            )

        return view_locations

    def _sample_start_poses(
        self,
        sim: Simulator,
        viewpoints: List,
    ) -> Tuple[List, List]:
        viewpoint_locs = [vp["agent_state"]["position"] for vp in viewpoints]
        start_positions = []
        start_rotations = []
        geodesic_distances = []
        euclidean_distances = []
        FAILED_TUPLE = ([], [], [], [])

        while len(start_positions) < self.start_poses_per_obj:
            for _ in range(self.start_retries):
                start_position = (
                    sim.pathfinder.get_random_navigable_point().astype(
                        np.float32
                    )
                )
                if (
                    start_position is None
                    or np.any(np.isnan(start_position))
                    or not sim.pathfinder.is_navigable(start_position)
                ):
                    raise RuntimeError("Unable to find valid starting location")

                # point should be not be isolated to a small poly island
                if (
                    sim.pathfinder.island_radius(start_position)
                    < self.ISLAND_RADIUS_LIMIT
                ):
                    continue

                geo_dist, closest_pt = self._geodesic_distance(
                    sim, start_position, viewpoint_locs
                )

                if not np.isfinite(geo_dist):
                    continue

                if (
                    geo_dist < self.start_distance_limits[0]
                    or geo_dist > self.start_distance_limits[1]
                ):
                    continue

                euc_dist = np.linalg.norm(start_position - closest_pt).item()
                dist_ratio = geo_dist / euc_dist
                if dist_ratio < self.min_geo_to_euc_ratio:
                    continue

                # aggressive _ratio_sample_rate (copied from PointNav)
                if np.random.rand() > (20 * (dist_ratio - 0.98) ** 2):
                    continue

                # check that the shortest path points are all on the same floor
                path = habitat_sim.ShortestPath()
                path.requested_start = start_position
                path.requested_end = closest_pt
                found_path = sim.pathfinder.find_path(path)
                if not found_path:
                    continue
                heights = [p.tolist()[1] for p in path.points]
                h_delta = max(heights) - min(heights)
                if h_delta > self.single_floor_threshold:
                    continue

                angle = np.random.uniform(0, 2 * np.pi)
                source_rotation = [
                    0,
                    np.sin(angle / 2),
                    0,
                    np.cos(angle / 2),
                ]  # Pick random starting rotation

                start_positions.append(start_position.tolist())
                start_rotations.append(source_rotation)
                geodesic_distances.append(geo_dist)
                euclidean_distances.append(euc_dist)
                break

            else:
                # no start pose found after n attempts
                return FAILED_TUPLE

        return (
            start_positions,
            start_rotations,
            geodesic_distances,
            euclidean_distances,
        )

    def _make_goal(
        self,
        sim: Simulator,
        pose_sampler: PoseSampler,
        obj: SemanticObject,
        with_viewpoints: bool,
        with_start_poses: bool,
    ):
        def log(s):
            log_text(self.verbose, self.scene, obj, obj_cat, s)

        obj_cat = self.cat_map[obj.category.name()]

        if with_start_poses:
            assert with_viewpoints

        states, hfovs = pose_sampler.sample_poses(obj)
        observations = self._render_poses(sim, states, hfovs)
        log(f"Poses rendered: {len(observations)}")

        observations, states, hfovs = self._can_see_object(
            observations, states, hfovs, obj
        )
        log(f"Poses can see object: {len(observations)}")

        if len(observations) == 0:
            log_text(self.verbose, self.scene, obj, obj_cat, "Image Goals: 0")
            return None

        frame_coverages = self._compute_frame_coverage(
            observations, obj.semantic_id
        )
        object_coverages, obj_surface_area = self._compute_object_coverage(
            states,
            [np.deg2rad(h) for h in hfovs],
            observations,
            obj,
            voxel_size=self.voxel_size,
            dbscan_slack=self.dbscan_slack,
        )

        if self.keep_metadata:
            self.metadata.append(
                {
                    "object_coverage": object_coverages,
                    "frame_coverage": frame_coverages,
                    "object_surface_area": obj_surface_area,
                    "poses_visible": len(object_coverages),
                    "obj_cat": obj.category.name(),
                    "obj_id": obj.id,
                    "has_episodes": False,
                }
            )

        keep_goal = self._threshold_image_goals(
            frame_coverages, object_coverages, obj_surface_area
        )
        log(f"Image Goals: {sum(keep_goal)}")

        if self.verbose:
            save_candidate_imgs(
                self.scene,
                obj,
                observations,
                frame_coverages,
                object_coverages,
                hfovs,
                keep_goal,
            )

        if sum(keep_goal) == 0:
            return None

        result = {
            "position": obj.aabb.center.tolist(),
            "object_id": obj.semantic_id,
            "object_name": obj.id,
            "object_category": obj_cat,
            "object_surface_area": obj_surface_area,
            "view_points": [],
            "image_goals": [],
            "start_positions": [],
            "start_rotations": [],
        }

        for s, h, fc, oc, k in zip(
            states,
            hfovs,
            frame_coverages,
            object_coverages,
            keep_goal,
        ):
            if k:
                ss = s.sensor_states["color_sensor"]
                result["image_goals"].append(
                    {
                        "position": ss.position.tolist(),
                        "rotation": quat_to_coeffs(ss.rotation).tolist(),
                        "frame_coverage": fc,
                        "object_coverage": oc,
                        "hfov": h,
                        "image_dimensions": list(self.img_size),
                    }
                )

        if not with_viewpoints:
            return result

        result["view_points"] = self._make_object_viewpoints(sim, obj)
        if len(result["view_points"]) == 0:
            return None

        if not with_start_poses:
            return result

        (
            result["start_positions"],
            result["start_rotations"],
            result["geodesic_distances"],
            result["euclidean_distances"],
        ) = self._sample_start_poses(sim, result["view_points"])
        if len(result["start_positions"]):
            if self.keep_metadata:
                self.metadata[-1]["has_episodes"] = True

            return result
        else:
            return None

    def make_image_goals(
        self,
        scene: str,
        with_viewpoints: bool,
        with_start_poses: bool,
    ) -> List[Dict[str, Any]]:
        self.scene = scene
        clear_log(self.verbose, scene)

        sim = self._config_sim(scene)
        pose_sampler = PoseSampler(sim=sim, **self.pose_sampler_args)

        objects = [
            o
            for o in sim.semantic_scene.objects
            if self.cat_map[o.category.name()] is not None
        ]

        image_goals = []
        for obj in objects:
            g = self._make_goal(
                sim, pose_sampler, obj, with_viewpoints, with_start_poses
            )
            if g is not None:
                image_goals.append(g)

        sim.close()
        self.scene = None
        return image_goals

    def goals_to_dataset(
        self,
        image_goals: Dict,
        scene: str,
        scene_config: str,
    ):
        data = {"goals": {}, "episodes": []}
        episode_id = 0
        for g in image_goals:
            goal_image_id = 0
            for sp, sr, geo, euc in zip(
                g["start_positions"],
                g["start_rotations"],
                g["geodesic_distances"],
                g["euclidean_distances"],
            ):
                data["episodes"].append(
                    {
                        "episode_id": str(episode_id),
                        "scene_id": scene,
                        "scene_dataset_config": scene_config,
                        "additional_obj_config_paths": [],
                        "start_position": sp,
                        "start_rotation": sr,
                        "info": {
                            "geodesic_distance": geo,
                            "euclidean_distance": euc,
                        },
                        "goals": [],
                        "object_category": g["object_category"],
                        "goal_object_id": str(g["object_id"]),
                        "goal_image_id": goal_image_id,
                    }
                )
                episode_id += 1
                goal_image_id = (goal_image_id + 1) % len(g["image_goals"])

            del g["start_positions"]
            del g["start_rotations"]
            del g["geodesic_distances"]
            del g["euclidean_distances"]

            short_scene = scene.split("/")[-1].split(".")[0]
            goal_key = f"{short_scene}_{g['object_id']}"
            data["goals"][goal_key] = g

        return data

    def save_metadata(self, save_to: str):
        if self.metadata is None:
            return
        import pickle

        with open(save_to, "wb") as f:
            pickle.dump(self.metadata, f)

    @staticmethod
    def _set_sensor_hfov(sim: Simulator, hfov: float) -> None:
        """override the HFOV of all sensors."""
        agent = sim.agents[0]
        for spec in agent.agent_config.sensor_specifications:
            if spec.hfov == hfov:
                continue

            spec.hfov = hfov
            hsim.SensorFactory.delete_subtree_sensor(
                agent.scene_node, spec.uuid
            )
            sensor_suite = hsim.SensorFactory.create_sensors(
                agent.scene_node, [spec]
            )
            agent._sensors[spec.uuid] = sensor_suite[spec.uuid]
            sim._update_simulator_sensors(spec.uuid, 0)

    @staticmethod
    def _render_poses(
        sim: Simulator, agent_states: List[AgentState], hfovs: List[float]
    ) -> List[Dict[str, ndarray]]:
        obs = []
        for agent_state, hfov in zip(agent_states, hfovs):
            ImageGoalGenerator._set_sensor_hfov(sim, hfov)
            sim.agents[0].set_state(agent_state, infer_sensor_states=False)
            obs.append(sim.get_sensor_observations())
        return obs

    @staticmethod
    def _can_see_object(
        observations: List[Dict[str, ndarray]],
        states: List[AgentState],
        hfovs: List[int],
        obj: SemanticObject,
    ):
        """Keep observations and sim states that can see the object."""
        keep_o, keep_s, keep_h = [], [], []
        for o, s, h in zip(observations, states, hfovs):
            if np.isin(obj.semantic_id, o["semantic_sensor"]):
                keep_o.append(o)
                keep_s.append(s)
                keep_h.append(h)

        return keep_o, keep_s, keep_h

    @staticmethod
    def _compute_frame_coverage(obs: List[Dict[str, ndarray]], oid: int):
        def _single(obs, oid):
            mask = obs["semantic_sensor"] == oid
            return (mask.sum() / mask.size).item()

        if isinstance(obs, list):
            return [_single(o, oid) for o in obs]
        if isinstance(obs, dict):
            return _single(obs, oid)
        else:
            raise TypeError("argument `obs` must be either a list or a dict.")

    @staticmethod
    def _compute_object_coverage(
        states: List[AgentState],
        hfovs: List[float],
        observations: List[Dict[str, ndarray]],
        obj: SemanticObject,
        voxel_size: float,
        dbscan_slack: float,
    ) -> Tuple[List[float], float]:
        camera_poses: List[SixDOFPose] = [
            s.sensor_states["semantic_sensor"] for s in states
        ]
        points = project_semantics_to_world(
            observations, camera_poses, hfovs, obj.semantic_id
        )

        obj_sa, obj_mesh = points_to_surface_area(
            np.concatenate(points, axis=1), obj.aabb, voxel_size, dbscan_slack
        )
        if obj_sa == 0.0:
            return [0.0 for _ in range(len(points))], 0.0

        scene = mesh_to_scene(obj_mesh)
        frame_sa = [
            points_to_surface_area_in_scene(
                f_pts, scene, voxel_size, dbscan_slack
            )
            / obj_sa
            for f_pts in points
        ]
        return frame_sa, obj_sa

    @staticmethod
    def _geodesic_distance(
        sim: Simulator,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
    ) -> float:
        path = habitat_sim.MultiGoalShortestPath()
        if isinstance(position_b[0], (Sequence, np.ndarray)):
            path.requested_ends = np.array(position_b, dtype=np.float32)
        else:
            path.requested_ends = np.array(
                [np.array(position_b, dtype=np.float32)]
            )
        path.requested_start = np.array(position_a, dtype=np.float32)
        sim.pathfinder.find_path(path)
        end_pt = path.points[-1] if len(path.points) else np.array([])
        return path.geodesic_distance, end_pt

    @staticmethod
    def save_to_disk(dataset, scene: str, outpath: str):
        split = os.path.dirname(scene).split("/")[-2]

        # save the split file if not exists
        split_file = os.path.join(outpath, split, f"{split}.json.gz")
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        if not os.path.exists(split_file):
            with gzip.open(split_file, "wt") as f:
                f.write(json.dumps({"episodes": []}))

        # save the scene content file
        short_scene_name = os.path.basename(scene).split(".")[0]
        scene_file = os.path.join(
            outpath, split, "content", f"{short_scene_name}.json.gz"
        )
        os.makedirs(os.path.dirname(scene_file), exist_ok=True)
        with gzip.open(scene_file, "wt") as f:
            f.write(json.dumps(dataset))


def make_scene_episodes(
    scene: Union[str, Tuple[str, str]],
    outpath: str,
    hm3d_location: str,
    save_metadata: bool,
):
    x1, y1 = (0.0, 0.02)
    x2, y2 = (25.0, 0.6)
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m * x2
    frame_cov_thresh_line = (m, b)

    scene_dataset_config = os.path.join(
        hm3d_location, "hm3d_annotated_basis.scene_dataset_config.json"
    )
    category_mapping_file = os.path.join(
        "goat/dataset/source_data", "Mp3d_category_mapping.tsv"
    )
    categories = {
        "chair": 0,
        "bed": 1,
        "plant": 2,
        "toilet": 3,
        "tv_monitor": 4,
        "sofa": 5,
    }
    iig_maker = ImageGoalGenerator(
        semantic_spec_filepath=scene_dataset_config,
        img_size=(512, 512),
        agent_height=1.41,  # stretch embodiment https://hello-robot.com/product
        agent_radius=0.17,
        pose_sampler_args={
            "r_min": 0.5,
            "r_max": 2.0,
            "r_step": 0.5,
            "rot_deg_delta": 10.0,
            "h_min": 0.8,
            "h_max": 1.5,
            "hfov_min": 60,
            "hfov_max": 120,
            "cameras_per_agent_pose": 5,
            "sample_lookat_deg_delta": 5.0,
        },
        category_mapping_file=category_mapping_file,
        categories=categories.keys(),
        min_object_coverage=0.7,
        frame_cov_thresh_line=frame_cov_thresh_line,
        voxel_size=0.05,
        dbscan_slack=0.01,
        goal_vp_cell_size=0.25,
        goal_vp_max_dist=1.0,
        start_poses_per_obj=2000,
        start_poses_tilt_angle=30.0,
        start_distance_limits=(1.0, 30.0),
        min_geo_to_euc_ratio=1.05,
        start_retries=2000,
        single_floor_threshold=0.25,
        keep_metadata=save_metadata,
        verbose=False,
    )

    image_goals = iig_maker.make_image_goals(
        scene=scene, with_viewpoints=True, with_start_poses=True
    )

    FRONT_STRIP = "/".join(hm3d_location.rstrip("/").split("/")[:-1])
    if scene.startswith(FRONT_STRIP):
        scene = scene[len(FRONT_STRIP) :]

    dataset_dict = iig_maker.goals_to_dataset(
        image_goals,
        scene,
        scene_config=f"./{scene_dataset_config}",
    )
    iig_maker.save_to_disk(dataset_dict, scene, outpath)

    if save_metadata:
        # save metadata to disk for object analysis plots
        split = os.path.dirname(scene).split("/")[-2]
        short_scene = scene.split("/")[-1].split(".")[0]
        metadata_f = os.path.join(
            outpath, "metadata", split, f"{short_scene}.pkl"
        )
        os.makedirs(os.path.dirname(metadata_f), exist_ok=True)
        iig_maker.save_metadata(metadata_f)


def make_episodes_for_split(
    scenes: List[str],
    outpath: str,
    save_metadata: bool,
    hm3d_location: str = "data/scene_datasets/hm3d/",
):
    for scene in tqdm(scenes, total=len(scenes), dynamic_ncols=True):
        make_scene_episodes(scene, outpath, hm3d_location, save_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/datasets/iin/hm3d/v1/",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--multiprocessing",
        dest="enable_multiprocessing",
        action="store_true",
    )
    parser.add_argument(
        "--start-poses-per-object",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--episodes-per-object",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--disable-euc-geo-ratio-check",
        action="store_true",
        dest="disable_euc_to_geo_ratio_check",
    )
    parser.add_argument(
        "--disable-wordnet-mapping",
        action="store_true",
        dest="disable_wordnet_mapping",
    )

    args = parser.parse_args()
    scenes = None
    if args.scene is not None:
        scene_id = args.scene.split(".")[0] + ".basis.glb"
        scenes = [scene_id]
    else:
        split = args.split.split("_")[0]
        scenes = list(
            get_hm3d_semantic_scenes("data/scene_datasets/hm3d", [split])[split]
        )
        scenes = sorted(scenes)

    if args.num_scenes > 0:
        scenes = scenes[: args.num_scenes]
    print(scenes)
    print(
        "Start poses per object: {}, Episodes per object: {}, Split: {}".format(
            args.start_poses_per_object, args.episodes_per_object, args.split
        )
    )

    # outpath = os.path.join(args.output_path, "{}/content/".format(args.split))

    make_episodes_for_split(scenes, args.output_path, True)
