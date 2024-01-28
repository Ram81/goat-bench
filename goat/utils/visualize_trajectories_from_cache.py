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
from goat.utils.utils import load_json


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


def preprocess_and_load_action_cache(path):
    cache = load_json(path)
    cache_json = {}
    for record in cache:
        scene_id = record["scene_id"].split("/")[-1].split(".")[0]
        cache_json["{}_{}".format(scene_id, record["episode_id"])] = record
    return cache_json


def get_next_action(episode, cache, current_step: int = 0):
    scene_id = episode.scene_id.split("/")[-1].split(".")[0]
    episode_id = episode.episode_id

    return cache["{}_{}".format(scene_id, episode_id)]["actions"][current_step]


def episode_in_cache(episode, cache):
    scene_id = episode.scene_id.split("/")[-1].split(".")[0]
    episode_id = episode.episode_id

    out = cache.get("{}_{}".format(scene_id, episode_id), None)
    return out


def concat_goal(env, current_subtask, goals, frame, observations, info):
    if current_subtask[1] == "object":
        frame = append_text_to_image(
            frame, "ObjectGoal: {}".format(current_subtask[0])
        )
    elif current_subtask[1] == "description":
        goal_description = goals[0]["lang_desc"]
        frame = append_text_to_image(
            frame, "LangGoal: {}".format(goal_description)
        )
        print("LangGoal: {}".format(goal_description))
    else:
        # img_goal = env.task.sensor_suite.sensors[
        #     "instance_imagegoal"
        # ]._get_instance_image_goal(img_goal)
        frame = observations_to_image({"rgb": observations["rgb"]}, info)

    return frame


def generate_trajectories(cfg, cache_path, video_dir="", num_episodes=1):
    os.makedirs(video_dir, exist_ok=True)
    with habitat.Env(cfg) as env:
        goal_radius = 0.1
        spl = defaultdict(float)
        softspl = defaultdict(float)
        total_success = defaultdict(float)
        total_episodes = 0.0

        cache = preprocess_and_load_action_cache(cache_path)

        logger.info("Total episodes: {}".format(len(env.episodes)))
        num_episodes = min(len(env.episodes), num_episodes)
        for episode_id in tqdm(range(num_episodes)):
            env.reset()
            success = 0
            current_step = 0

            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = 1.41
            navmesh_settings.agent_radius = 0.17
            navmesh_settings.agent_max_climb = 0.1
            navmesh_settings.cell_height = 0.05
            navmesh_success = env.sim.recompute_navmesh(
                env.sim.pathfinder,
                navmesh_settings,
                include_static_objects=False,
            )
            print("Nav recompute success: {}".format(navmesh_success))

            episode = env.current_episode
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]

            info = {}
            obs_list = [[]]
            goals = []

            episode_meta = episode_in_cache(episode, cache)
            if episode_meta is None:
                continue
            # print(episode_meta["actions"])

            for k, v in episode_meta.items():
                if "actions" in k:
                    continue
                print(k, v)
            print(
                "Current: {}- {} - {}".format(
                    episode.episode_id, scene_id, episode.tasks
                )
            )
            subtask_idx = 0

            while not env.episode_over:
                best_action = get_next_action(episode, cache, current_step)

                observations = env.step(best_action)
                current_step += 1
                prev_info = info

                info = env.get_metrics()

                frame = observations_to_image(
                    {"rgb": observations["rgb"]}, info
                )

                if best_action == 6:
                    print(
                        "[Sub: {}] Action: {} - {}\n\n".format(
                            subtask_idx,
                            best_action,
                            {
                                k: v
                                for k, v in prev_info.items()
                                if "top_down_map" not in k
                            },
                        )
                    )
                    subtask_idx += 1
                    obs_list.append([])

                if len(episode.tasks) != env.task.active_subtask_idx:
                    frame = concat_goal(
                        env,
                        episode.tasks[env.task.active_subtask_idx],
                        episode.goals[env.task.active_subtask_idx],
                        frame,
                        observations,
                        info,
                    )
                else:
                    frame = append_text_to_image(frame, "End")
                obs_list[-1].append(frame)

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

            print(episode_meta.keys())
            print(
                "Num subtasks: {} - {} - {}".format(
                    len(episode.tasks),
                    len(obs_list),
                    episode_meta["success_by_subtask"],
                )
            )

            out_path = os.path.join(
                video_dir, "{}/ep_{}".format(scene_id, episode.episode_id)
            )
            os.makedirs(out_path, exist_ok=True)

            for idx, obs in enumerate(obs_list[:-1]):
                print(":Num steps: {}".format(len(obs)))
                make_videos(
                    [obs],
                    out_path,
                    "subtask_{}_success={}".format(
                        idx,
                        info["success"]["subtask_success"][idx],
                    ),
                )
            break
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
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--scene", type=str, default="")
    args = parser.parse_args()

    objectnav_config = "config/tasks/goat_stretch_hm3d.yaml"
    config = get_config(objectnav_config)
    with read_write(config):
        config.habitat.simulator.type = "OVONSim-v0"
        config.habitat.dataset.type = "Goat-v1"
        config.habitat.dataset.split = "val_seen"
        config.habitat.dataset.scenes_dir = "data/scene_datasets/hm3d/"
        config.habitat.dataset.content_scenes = [args.scene]
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
        config.habitat.task.measurements.top_down_map = GoatTopDownMapConfig()
        # del config.habitat.task.measurements["distance_to_goal_reward"]
        config.habitat.task.measurements[
            "goat_distance_to_goal_reward"
        ] = GoatDistanceToGoalRewardConfig()
        config.habitat.task.measurements.success.success_distance = 0.25

    generate_trajectories(
        config,
        args.cache,
        video_dir=args.video_dir,
        num_episodes=args.num_episodes,
    )


if __name__ == "__main__":
    main()
