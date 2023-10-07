import argparse
import glob
import json
import os

import habitat
import numpy as np
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from goat.models.encoders.vc1 import VC1Encoder
from goat.utils.utils import save_image, save_pickle


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
        self.init_visual_encoder()

    def init_visual_encoder(self):
        self.encoder = VC1Encoder(device=self.device)

    def config_env(self, scene):
        config = get_config(self.config_path)
        with read_write(config):
            config.habitat.dataset.data_path = os.path.join(
                self.data_path, f"{self.split}/{self.split}.json.gz"
            )
            config.habitat.dataset.content_scenes = [scene]

        env = habitat.Env(config=config)
        return env

    def run(self, scene):
        data = {}
        env = self.config_env(scene)
        env.reset()
        goals = env._dataset.goals

        print("Scene reset: {}".format(scene))
        os.makedirs(self.output_path, exist_ok=True)

        for goal_k, goal_val in goals.items():
            print("Goal: {} - {}".format(goal_k, len(goal_val.image_goals)))
            goals_meta = []
            for goal_idx, img_goal in enumerate(goal_val.image_goals):
                # Embedding directory and files
                vc1_file = f"vc1_embedding_{goal_idx}.npy"

                img = env.task.sensor_suite.sensors[
                    "instance_imagegoal"
                ]._get_instance_image_goal(img_goal)

                vc1_embedding = self.encoder.embed_vision(img)
                metadata = dict(
                    hfov=img_goal.hfov,
                    object_id=goal_val.object_id,
                    position=img_goal.position,
                    rotation=img_goal.rotation,
                    goal_id=goal_idx,
                    embedding=vc1_embedding,
                )
                goals_meta.append(metadata)

            data[f"{goal_k}"] = goals_meta

        out_path = os.path.join(self.output_path, f"{scene}_embedding.pkl")
        save_pickle(data, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/instance_imagenav_v2/ver_ovrl_instance_imagenav.yaml",
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
        "--scene",
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
    cache.run(args.scene)
