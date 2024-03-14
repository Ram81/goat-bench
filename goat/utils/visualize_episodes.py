import argparse
import os
from collections import defaultdict

import habitat
import habitat_sim
import numpy as np
from habitat import get_config, logger
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    images_to_video,
    observations_to_image,
)
from habitat_sim.utils.common import quat_from_two_vectors
from numpy import ndarray
from tqdm import tqdm

from goat.config import (
    ClipObjectGoalSensorConfig,
    GoatDistanceToGoalConfig,
    GoatDistanceToGoalRewardConfig,
    GoatSoftSPLConfig,
    GoatSPLConfig,
    GoatSuccessConfig,
    GoatTopDownMapConfig,
)
from goat.dataset import goat_dataset, ovon_dataset
from goat.utils.utils import save_image


def _face_object(object_position: np.array, point: ndarray):
    EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
    cam_normal = (object_position - point) + EPS_ARRAY
    cam_normal[1] = 0
    cam_normal = cam_normal / np.linalg.norm(cam_normal)
    return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)


def make_videos(observations_list, output_dir, id):
    images_to_video(observations_list[0], output_dir=output_dir, video_name=id)


def get_nearest_goal(episode, env):
    min_dist = 1000.0
    sim = env.sim

    goals = episode.goals[env.task.active_subtask_idx]

    goal_location = None
    goal_rotation = None

    agent_position = sim.get_agent_state().position
    for goal in goals:
        # for goal in goal_obj:
        for view_point in goal["view_points"]:
            position = view_point["agent_state"]["position"]

            dist = sim.geodesic_distance(agent_position, position)
            if min_dist > dist:
                min_dist = dist
                goal_location = position
                goal_rotation = view_point["agent_state"]["rotation"]
    return goal_location, goal_rotation


def generate_trajectories(cfg, output_dir="", num_episodes=1):
    os.makedirs(output_dir, exist_ok=True)
    with habitat.Env(cfg) as env:
        logger.info("Total episodes: {}".format(len(env.episodes)))
        total_success = defaultdict(float)

        print("Action space: {}".format(env.action_space))

        num_episodes = min(len(env.episodes), num_episodes)
        for episode_id in tqdm(range(num_episodes)):
            observations = env.reset()

            observations = env.step(6)

            episode = env.current_episode
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]

            # _ = env.step(6)
            info = env.get_metrics()

            for k, v in info["success"].items():
                if isinstance(v, list):
                    continue
                total_success[k] += v
            frame = observations_to_image({"rgb": observations["rgb"]}, info)

            os.makedirs(os.path.join(output_dir, scene_id), exist_ok=True)
            save_image(
                frame,
                os.path.join(output_dir, scene_id, f"ep_{episode_id}.png"),
            )
        print("Total success: {}".format(total_success))
        print(
            "Mean success: {}".format(
                {k: v / num_episodes for k, v in total_success.items()}
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/episodes/sampled.json.gz"
    )
    parser.add_argument("--output-dir", type=str, default="data/video_dir/")
    parser.add_argument("--num-episodes", type=int, default=2)
    args = parser.parse_args()

    objectnav_config = "config/tasks/goat_stretch_hm3d.yaml"
    config = get_config(objectnav_config)
    with read_write(config):
        config.habitat.dataset.type = "Goat-v1"
        config.habitat.dataset.split = "train"
        config.habitat.dataset.scenes_dir = "data/scene_datasets/hm3d/"
        config.habitat.dataset.content_scenes = ["*"]
        config.habitat.dataset.data_path = args.data
        del config.habitat.task.lab_sensors["objectgoal_sensor"]
        config.habitat.task.measurements.distance_to_goal = (
            GoatDistanceToGoalConfig()
        )
        del config.habitat.task.measurements["soft_spl"]
        del config.habitat.task.measurements["distance_to_goal_reward"]
        config.habitat.task.measurements.success = GoatSuccessConfig()
        config.habitat.task.measurements.spl = GoatSPLConfig()
        config.habitat.task.measurements.soft_spl = GoatSoftSPLConfig()
        print(config.habitat.task.measurements)
        config.habitat.task.measurements.top_down_map = GoatTopDownMapConfig()
        # del config.habitat.task.measurements["distance_to_goal_reward"]
        config.habitat.task.measurements[
            "goat_distance_to_goal_reward"
        ] = GoatDistanceToGoalRewardConfig()
        config.habitat.task.measurements.success.success_distance = 0.25

    generate_trajectories(
        config, output_dir=args.output_dir, num_episodes=args.num_episodes
    )


if __name__ == "__main__":
    main()