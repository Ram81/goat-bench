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

import frontier_exploration
from frontier_exploration.measurements import (
    FrontierExplorationMapMeasurementConfig,
)
from frontier_exploration.objnav_explorer import ObjNavExplorerSensorConfig
from goat_bench.config import (
    ClipObjectGoalSensorConfig,
    GoatDistanceToGoalConfig,
    GoatDistanceToGoalRewardConfig,
    GoatSoftSPLConfig,
    GoatSPLConfig,
    GoatSuccessConfig,
    GoatTopDownMapConfig,
)
from goat_bench.dataset import goat_dataset, ovon_dataset


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


def generate_trajectories(cfg, video_dir="", num_episodes=1):
    os.makedirs(video_dir, exist_ok=True)
    with habitat.Env(cfg) as env:
        goal_radius = 0.1
        spl = defaultdict(float)
        softspl = defaultdict(float)
        total_success = defaultdict(float)
        total_episodes = 0.0
        scene_id = env._current_episode.scene_id.split("/")[-1].split(".")[0]

        logger.info("Total episodes: {}".format(len(env.episodes)))
        num_episodes = min(len(env.episodes), num_episodes)
        for episode_id in tqdm(range(num_episodes)):
            follower = ShortestPathFollower(env._sim, goal_radius, False)
            observations = env.reset()
            success = 0
            episode = env.current_episode
            goal_position, goal_rotation = get_nearest_goal(episode, env)

            info = {}
            obs_list = []
            if goal_position is None:
                continue
            active_subtask_idx = env.task.active_subtask_idx
            logger.info(
                "Subtask start {}".format(
                    active_subtask_idx,
                )
            )
            steps_per_subtask = []
            task_reset_prev_step = False
            steps = 0

            while not env.episode_over:
                goal_position, goal_rotation = get_nearest_goal(episode, env)
                best_action = observations["objnav_explorer"][0]

                pre_metrics = env.get_metrics()

                if best_action == 6:
                    logger.info(
                        "Step {} - {} - {}".format(
                            steps, best_action, env.task.active_subtask_idx
                        )
                    )

                observations = env.step(best_action)
                steps_per_subtask.append(best_action)

                info = env.get_metrics()

                if active_subtask_idx != env.task.active_subtask_idx:
                    active_subtask_idx = env.task.active_subtask_idx
                    logger.info(
                        "Subtask stop {} - {} - {} - {} - {} - {} - {}\n".format(
                            best_action,
                            pre_metrics["distance_to_goal"],
                            info["distance_to_goal"],
                            env.task.active_subtask_idx,
                            len(episode.tasks),
                            steps_per_subtask,
                            observations["objnav_explorer"],
                        )
                    )
                    steps_per_subtask = []

                if best_action == HabitatSimActions.stop:
                    position = env.sim.get_agent_state().position
                    observations = env.sim.get_observations_at(
                        position, goal_rotation, False
                    )

                frame = observations_to_image(
                    {"rgb": observations["rgb"]}, info
                )

                text = ""
                if len(episode.tasks) != env.task.active_subtask_idx:
                    text = "{} - {}".format(
                        episode.tasks[env.task.active_subtask_idx][0],
                        episode.tasks[env.task.active_subtask_idx][1],
                    )

                frame = append_text_to_image(
                    frame,
                    "Goal: {}".format(text),
                )
                obs_list.append(frame)
                steps += 1

                success = info["success"]

            for k, v in success.items():
                if isinstance(v, list):
                    continue
                total_success[k] += v

            for k, v in info["spl"].items():
                if isinstance(v, list):
                    continue
                spl[k] += v

            for k, v in info["soft_spl"].items():
                softspl[k] += v

            # spl += info["spl"]
            total_episodes += 1

            make_videos(
                [obs_list], video_dir, "{}_{}".format(scene_id, episode_id)
            )
        print("Total episodes: {}".format(total_episodes))

        print(
            "\n\nEpisode success: {}".format(
                {k: v / total_episodes for k, v in total_success.items()}
            )
        )
        print("SPL: {}, {}".format(spl, total_episodes))
        print("SoftSPL: {}, {}".format(softspl, total_episodes))
        print("Success: {}, {}".format(total_success, total_episodes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/episodes/sampled.json.gz"
    )
    parser.add_argument("--video-dir", type=str, default="data/video_dir/")
    parser.add_argument("--num-episodes", type=int, default=2)
    args = parser.parse_args()

    objectnav_config = "config/tasks/goat_stretch_hm3d.yaml"
    config = get_config(objectnav_config)
    with read_write(config):
        config.habitat.dataset.split = "train"
        config.habitat.dataset.scenes_dir = "data/scene_datasets/hm3d/"
        config.habitat.dataset.content_scenes = ["*"]
        config.habitat.dataset.data_path = args.data

        config.habitat.simulator.habitat_sim_v0.allow_sliding = True

        config.habitat.task.lab_sensors["objnav_explorer"] = (
            ObjNavExplorerSensorConfig()
        )
        config.habitat.task.measurements["frontier_exploration_map"] = (
            FrontierExplorationMapMeasurementConfig()
        )

    generate_trajectories(
        config, video_dir=args.video_dir, num_episodes=args.num_episodes
    )


if __name__ == "__main__":
    main()
