import argparse
import glob
import json
import os

import habitat
import numpy as np
import torch
from habitat.config import read_write
from habitat_baselines.config.default import get_config

from goat.models.encoders.clip import CLIPEncoder
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
        self.init_visual_encoder(encoder)
        self.encoder_name = encoder

    def init_visual_encoder(self, encoder):
        if encoder == "VC-1":
            self.encoder = VC1Encoder(device=self.device)
        elif encoder == "CLIP":
            self.encoder = CLIPEncoder(device=self.device)
        else:
            raise NotImplementedError

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
        data_goal = {}
        env = self.config_env(scene)
        env.reset()
        goals = env._dataset.goals

        print("Scene reset: {}".format(scene))
        os.makedirs(self.output_path, exist_ok=True)

        for goal_k, goal_val in goals.items():
            goals_meta = []
            for goal_idx, img_goal in enumerate(goal_val.image_goals):
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

            scene_id = goal_k.split("_")[0]
            data[f"{goal_k}"] = goals_meta
            data_goal[f"{scene_id}_{goal_val.object_name}"] = goals_meta

        # out_path = os.path.join(self.output_path, f"{scene}_clip_embedding.pkl")
        # save_pickle(data, out_path)

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_goat_embedding.pkl"
        )
        save_pickle(data_goal, out_path)


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
    parser.add_argument(
        "--encoder",
        type=str,
        default="VC-1",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        data_path=args.input_path,
        split=args.split,
        output_path=args.output_path,
        encoder=args.encoder,
    )
    cache.run(args.scene)
