# Reference: https://github.com/apple/ml-llarp/blob/main/llarp/policies/visual_encoders.py
from . import clip
from einops import rearrange
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import Normalize

from .base import VisualEncoder

import os
os.environ['CLIP_CACHE'] = './cache'
# options: 
class ClipEncoder(VisualEncoder):
    """
    Wrapper for CLIP visual encoder
    """
    def __init__(self, im_obs_space, use_b16: bool, checkpoint_name: str="ViT-B/32", linear_proj=True, **kwargs):
        super().__init__()
        if checkpoint_name not in clip.available_models():
            raise ValueError(f"Name: {checkpoint_name} not in CLIP available models.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_proj = linear_proj
        model, preprocess = clip.load(checkpoint_name, device=device, download_root=os.environ['CLIP_CACHE'], linear_proj=self.linear_proj)

        self.net = model
        if self.linear_proj:
            self.embd_size = model.visual.output_dim
        else:
            self.embd_size = model.visual.width

        preprocess_transforms = [
            # resize and center crop to 224
            preprocess.transforms[0],
            preprocess.transforms[1],
            T.ConvertImageDtype(torch.float),
            Normalize((0, 0, 0), (255, 255, 255)),
            # normalize with CLIP mean, std
            preprocess.transforms[4],
        ]
        self.model_transforms = T.Compose(preprocess_transforms)
        # TODO: Check if this classifier feature is needed
        # self.net.classifier_feature = classifier_feature
        self._image_obs_keys = im_obs_space.spaces.keys()

        if use_b16:
            self._use_type = torch.bfloat16
        else:
            self._use_type = torch.float32
            
        self.to(self._use_type)

    def forward(self, obs):
        img = torch.cat(
            [v for k, v in obs.items() if k in self._image_obs_keys], dim=-1
        )
        img = img.to(self._use_type)

        assert img.shape[-1] == 3, "CLIP cannot take a non-RGB image."
        img = self.model_transforms(img.permute(0, 3, 1, 2))
        ret = self.net.encode_image(img)

        if self.linear_proj:
            ret = rearrange(ret, "b d -> b 1 d")
        assert ret.shape[1:] == self.output_shape

        return ret.to(torch.float32)
    
    @property
    def output_shape(self):
        if not self.linear_proj:
            return (
                np.prod((
                    self.net.visual.input_resolution//self.net.visual.patch_size,
                    self.net.visual.input_resolution//self.net.visual.patch_size
                )), self.embd_size
            )
        else:
            return (1, self.embd_size)
