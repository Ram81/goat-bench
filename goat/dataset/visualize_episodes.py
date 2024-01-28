import argparse
import glob
import os

import cv2
import habitat
import habitat_sim
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    HabitatSimSemanticSensorConfig,
)
from habitat.utils.visualizations import maps
from tqdm import tqdm

from goat.utils.utils import (
    draw_bounding_box,
    draw_point,
    is_on_same_floor,
    load_dataset,
)

SCENES_ROOT = "data/scene_datasets/hm3d/"
MAP_RESOLUTION = 512


def get_objnav_config(scene):
    TASK_CFG = "config/tasks/objectnav_stretch_hm3d.yaml"
    SCENE_DATASET_CFG = os.path.join(
        SCENES_ROOT, "hm3d_annotated_basis.scene_dataset_config.json"
    )
    objnav_config = get_config(TASK_CFG)

    deviceId = 0

    with read_write(objnav_config):
        # TODO: find a better way to do it.
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = (
            HabitatSimSemanticSensorConfig()
        )
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = deviceId
        objnav_config.habitat.simulator.scene = "data/scene_datasets/" + scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG
        objnav_config.habitat.simulator.habitat_sim_v0.enable_physics = True

        objnav_config.habitat.task.measurements = {}

    return objnav_config


def get_sim(objnav_config):
    sim = habitat.sims.make_sim(
        "Sim-v0", config=objnav_config.habitat.simulator
    )

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    navmesh_settings.agent_max_climb = 0.1
    navmesh_settings.cell_height = 0.05
    sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=True
    )

    return sim


def setup(scene):
    print(scene)
    ovon_config = get_objnav_config(scene)
    sim = get_sim(ovon_config)
    return sim


def visualize_episodes(sim, episodes, scene_id):
    top_down_maps = []

    top_down_map = maps.get_topdown_map(
        sim.pathfinder,
        height=episodes[0]["start_position"][1],
        map_resolution=MAP_RESOLUTION,
        draw_border=True,
    )

    for episode in tqdm(episodes):
        draw_point(
            sim,
            top_down_map,
            episode["start_position"],
            maps.MAP_SOURCE_POINT_INDICATOR,
        )

    top_down_map = maps.colorize_topdown_map(top_down_map)
    top_down_maps.append(top_down_map)

    print("Top down maps: {}".format(len(top_down_maps)))

    return top_down_maps


def save_visual(img, path):
    cv2.imwrite(path, img)


def visualize(data_path, output_path):
    files = glob.glob(os.path.join(data_path, "*.json.gz"))
    os.makedirs(output_path, exist_ok=True)

    for idx, file in enumerate(files):
        scene_id = file.split("/")[-1].split(".")[0]
        dataset = load_dataset(file)
        print("[{}/{}] Scene id: {}".format(idx, len(files), scene_id))

        os.makedirs(output_path, exist_ok=True)

        sim = setup(dataset["episodes"][0]["scene_id"].replace("//", "/"))

        top_down_maps = visualize_episodes(
            sim,
            dataset["episodes"],
            scene_id,
        )
        scene_output_path = os.path.join(output_path, "{}.png".format(scene_id))
        save_visual(top_down_maps[0], scene_output_path)
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to episode dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of visualization",
    )
    args = parser.parse_args()
    visualize(args.dataset, args.output_path)
