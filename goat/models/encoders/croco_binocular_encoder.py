from typing import Dict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from goat.models.encoders.croco2_vit_encoder import Croco2ViTEncoder
from goat.models.encoders.croco2_vit_decoder import Croco2ViTDecoder
from torchvision import transforms as T
from gym import spaces


class CrocoBinocularEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        checkpoint: str,
        adapter: bool,
        hidden_size: int,
    ):
        super().__init__()
        
        self.goal_imagenav = "image_goal_rotation" in observation_space.spaces
        self.goal_instance_imagenav = "goat_instance_imagegoal" or "instance_imagegoal" in observation_space.spaces
        self.goal_features = "cache_croco_goal_feat" in observation_space.spaces and "cache_croco_goal_pos" in observation_space.spaces
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.preprocess = T.Compose([
            T.Resize(112, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(112),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(checkpoint)
        kwargs = ckpt.get('croco_kwargs', {})
        if 'img_size' in kwargs:
            del kwargs['img_size']
        self.encoder = Croco2ViTEncoder(adapter=adapter, img_size=112, **kwargs)
        self.decoder = Croco2ViTDecoder(adapter=adapter, img_size=112, **kwargs)
        encoder_weights = {k:v for k,v in ckpt['model'].items() if k.startswith('enc') or k.startswith('patch')}
        msg = self.encoder.load_state_dict(encoder_weights, strict=False)
        print(f"Loading Croco encoder weights from {checkpoint}: {msg}")
        decoder_weights = {k:v for k,v in ckpt['model'].items() if k.startswith('dec') or k.startswith('patch')}
        msg = self.decoder.load_state_dict(decoder_weights, strict=False)
        print(f"Loading Croco decoder weights from {checkpoint}: {msg}")

        print("Freezing Croco encoder parameters")
        for name, param in self.encoder.named_parameters():
            if "adaptmlp" not in name:
                param.requires_grad = False
        self.encoder.eval()
        print("Freezing Croco decoder parameters")
        for name, param in self.decoder.named_parameters():
            if "adaptmlp" not in name:
                param.requires_grad = False
        self.decoder.eval()

        dec_embed_dim = self.decoder.dec_embed_dim
        # TODO: Check if we should add linear here
        # OR if linear and flatten is okay at the goal_fc level
        # Original uses a single linear and flatten before passing to policy
        # TODO: Check if should add a ReLU here, not mentioned in the paper
        self.fc = nn.Sequential(
            nn.Linear(
                dec_embed_dim,
                hidden_size,
            ),
            nn.Flatten()
        )

    def forward(self, observations) -> torch.Tensor:  # type: ignore
        rgb = observations["rgb"]
        rgb = rgb.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
        rgb = self.preprocess(rgb)
        rgb_feat, rgb_pos = self.encoder(rgb)

        # NOTE: Do we need to handle the number of environments here in any way?
        if self.goal_instance_imagenav:
            instance_imagegoal = observations["instance_imagegoal"] if "instance_imagegoal" in observations else observations["goat_instance_imagegoal"]
            instance_imagegoal = instance_imagegoal.permute(0, 3, 1, 2)
            instance_imagegoal = self.preprocess(instance_imagegoal)
            goal_feat, goal_pos = self.encoder(instance_imagegoal)
        elif self.goal_imagenav:
            imagegoal = observations["image_goal_rotation"]
            imagegoal = imagegoal.permute(0, 3, 1, 2)
            imagegoal = self.preprocess(imagegoal)
            goal_feat, goal_pos = self.encoder(imagegoal)
        elif self.goal_features:
            goal_feat, goal_pos = observations["cache_croco_goal_feat"], observations["cache_croco_goal_pos"].long()

        else:
            raise NotImplementedError("Required observations not found.")

        x = self.decoder(rgb_feat, rgb_pos, goal_feat, goal_pos)
        
        x = self.fc(x)
        return x