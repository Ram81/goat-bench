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

from goat.models.encoders.clip import CLIPEncoder
from goat.models.encoders.vc1 import VC1Encoder
from goal.models.encoders.croco2_encoder import Croco2Encoder
from goat.utils.utils import save_image, save_pickle


class CacheGoals:
    def __init__(
        self,
        config_path: str,
        data_path: str = "",
        split: str = "train",
        output_path: str = "",
        encoder: str = "VC-1",
        add_noise: bool = False,
    ) -> None:
        self.device = torch.device("cuda")

        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.split = split
        self.init_visual_encoder(encoder)
        self.encoder_name = encoder
        self.add_noise = add_noise

    def init_visual_encoder(self, encoder):
        if "VC-1" in encoder:
            self.encoder = VC1Encoder(device=self.device)
        elif "CLIP" in encoder:
            self.encoder = CLIPEncoder(device=self.device)
        elif encoder == "CroCo-V2":
            self.encoder = Croco2Encoder(device=self.device)
        else:
            raise NotImplementedError

    def apply_noise(self, image):
        mean = 0
        std = random.uniform(0.1, 2.0)
        image = image + np.random.normal(
            loc=mean, scale=std, size=image.shape
        ).astype(np.float32)
        return image.astype(np.uint8)

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
        if os.path.exists(
            os.path.join(
                self.output_path,
                f"{scene}_{self.encoder_name}_iin_embedding.pkl",
            )
        ):
            print("Scene already cached: {}".format(scene))
            return

        data = {}
        data_goal = {}
        env = self.config_env(scene)
        env.reset()
        goals = env._dataset.goals

        print("Scene reset: {}".format(scene))
        os.makedirs(self.output_path, exist_ok=True)

        print("Add noise: {}".format(self.add_noise))

        for goal_k, goal_val in goals.items():
            goals_meta = []
            for goal_idx, img_goal in enumerate(goal_val.image_goals):
                img = env.task.sensor_suite.sensors[
                        "instance_imagegoal"
                    ]._get_instance_image_goal(img_goal)

                if self.add_noise:
                    img = self.apply_noise(img)

                if self.encoder_type == "VC-1":
                    # vc1_file = f"vc1_embedding_{goal_idx}.npy"
                    vc1_embedding = self.encoder.embed_vision(img)
                
                    metadata = dict(
                        hfov=img_goal.hfov,
                        object_id=goal_val.object_id,
                        position=img_goal.position,
                        rotation=img_goal.rotation,
                        goal_id=goal_idx,
                        embedding=vc1_embedding,
                    )

                elif self.encoder_type == "CroCo-V2":
                    # croco_file = f"croco_embedding_{goal_idx}.npy"
                    croco_embedding = self.encoder.embed_vision(img)
                    
                    metadata = dict(
                        hfov=img_goal.hfov,
                        object_id=goal_val.object_id,
                        position=img_goal.position,
                        rotation=img_goal.rotation,
                        goal_id=goal_idx,
                        embedding=croco_embedding,
                    )
                goals_meta.append(metadata)

            scene_id = goal_k.split("_")[0]
            data[f"{goal_k}"] = goals_meta
            data_goal[f"{scene_id}_{goal_val.object_name}"] = goals_meta

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_iin_embedding.pkl"
        )
        save_pickle(data, out_path)

        out_path = os.path.join(
            self.output_path, f"{scene}_{self.encoder_name}_goat_embedding.pkl"
        )
        save_pickle(data_goal, out_path)


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
    parser.add_argument(
        "--add-noise",
        action="store_true",
        dest="add_noise",
    )
    args = parser.parse_args()

    cache = CacheGoals(
        config_path=args.config,
        data_path=args.input_path,
        split=args.split,
        output_path=args.output_path,
        encoder=args.encoder,
        add_noise=args.add_noise,
    )
    cache.run(args.scene)
