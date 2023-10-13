from typing import Any, Optional

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from goat.models.encoders.croco2_vit_encoder import Croco2ViTEncoder
from goat.models.encoders.croco2_vit_decoder import Croco2ViTDecoder
from torchvision.transforms import ToTensor, Normalize, Compose


class CrocoBinocularEncoder(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        hidden_size: int,
        visual_transform: Any = None,
    ):
        super().__init__()
        
        self.visual_transform = visual_transform
        ckpt = torch.load(checkpoint, 'cpu')
        self.encoder = Croco2ViTEncoder(**ckpt.get('croco_kwargs', {}))
        self.decoder = Croco2ViTDecoder(**ckpt.get('croco_kwargs', {}))
        encoder_weights = {k:v for k,v in ckpt['model'].items() if k.startswith('enc') or k.startswith('patch')}
        msg = self.encoder.load_state_dict(encoder_weights)
        decoder_weights = {k:v for k,v in ckpt['model'].items() if k.startswith('dec') or k.startswith('patch')}
        msg = self.decoder.load_state_dict(decoder_weights)
        
        # TODO: Check if we should add linear here
        # OR if linear and flatten is okay at the goal_fc level
        # Original uses a single linear and flatten before passing to policy
        # TODO: Check if should add a ReLU here, not mentioned in the paper
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoder.output_size,
                hidden_size,
            )
        )

    def forward(self, observations: "TensorDict") -> torch.Tensor:  # type: ignore
        rgb = observations["rgb"]
        num_environments = rgb.size(0)
        rgb = self.visual_transform(rgb, num_environments)
        rgb_feat, rgb_pos = self.encoder(rgb)
        if "instance_imagegoal" in observations:
            instance_imagegoal = observations["instance_imagegoal"]
            instance_imagegoal = self.visual_transform(instance_imagegoal, num_environments)
            goal_feat, goal_pos = self.encoder(instance_imagegoal)

        elif "cache_croco_instance_imagegoal" in observations:
            # NOTE: Do we need to handle the number of environments here in any way?
            goal_feat, goal_pos = observations["cache_croco_instance_imagegoal"]

        else:
            raise NotImplementedError("Required observations not found.")
        
        x = self.decoder(rgb_feat, rgb_pos, goal_feat, goal_pos)
        
        x = self.fc(x)
        return x