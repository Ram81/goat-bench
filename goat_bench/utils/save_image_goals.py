import argparse
import glob
import json
import os
import random

import habitat
import numpy as np
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config
from tqdm import tqdm

from goat_bench.utils.utils import save_image, write_json


class CacheGoals:
    def __init__(
        self,
        config_path: str,
        data_path: str = "",
        split: str = "train",
        output_path: str = "",
        encoder: str = "VC-1",
    ) -> None:
        self.device = torch.device("cuda")

        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.split = split

    def config_env(self, scene):
        config = get_config(self.config_path)
        with read_write(config):
            config.habitat.dataset.data_path = os.path.join(
                self.data_path, f"{self.split}/{self.split}.json.gz"
            )
            config.habitat.dataset.content_scenes = [scene]

        env = habitat.Env(config=config)
        return env

    def run(self):
        scenes = glob.glob(
            os.path.join(self.data_path, self.split, "content/*.json.gz")
        )
        print("Total scenes: {}".format(len(scenes)))
        for scene in tqdm(scenes):
            scene = scene.split("/")[-1].split(".")[0]
            env = self.config_env(scene)
            env.reset()
            goals = env._dataset.goals

            print("Scene reset: {}".format(scene))
            output_path = os.path.join(self.output_path, self.split, scene)
            os.makedirs(output_path, exist_ok=True)
            records = []

            for goal_k, goal_val in goals.items():
                goals_meta = []

                img_goal = random.choice(goal_val.image_goals)
                img_goal.hfov = 120
                img_goal.dimensions = [512, 512]

                img = env.task.sensor_suite.sensors[
                    "instance_imagegoal"
                ]._get_instance_image_goal(img_goal)

                img_path = os.path.join(
                    output_path, f"{goal_k}_{goal_val.object_category}.png"
                )
                save_image(img, img_path)
                records.append(
                    {
                        "goal_id": goal_k,
                        "object_category": goal_val.object_category,
                        "scene": scene,
                        "img_path": img_path,
                    }
                )
            write_json(records, os.path.join(output_path, "records.json"))
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/tasks/instance_imagenav_stretch_hm3d.yaml",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        data_path=args.input_path,
        split=args.split,
        output_path=args.output_path,
    )
    cache.run()
