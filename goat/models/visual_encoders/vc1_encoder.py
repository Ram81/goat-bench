# Reference: https://github.com/apple/ml-llarp/blob/main/llarp/policies/visual_encoders.py
from einops import rearrange
import numpy as np
import torch
from vc_models.models.vit import model_utils

from .base import VisualEncoder


class Vc1Encoder(VisualEncoder):
    """
    Wrapper for the VC1 visual encoder using the CLS encoder.

    :param classifier_feature: Either "use_cls_token" or "reshape_embedding".
    """

    def __init__(self, im_obs_space, use_b16: bool, classifier_feature: str, checkpoint_name: str=model_utils.VC1_BASE_NAME, **kwargs):
        super().__init__()

        (
            self.net,
            self.embd_size,
            self.model_transforms,
            model_info,
        ) = model_utils.load_model(checkpoint_name)
        self.net.classifier_feature = classifier_feature

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

        # Image encoder expects shape [batch_size, img_width, img_height, img_channels]
        if len(img.shape) == 5:
            # We have a sequence dimension as well. Flatten that into the batch dimension
            expand_shape = img.shape[:2]
            img = rearrange(img, "b t w h c -> (b t) w h c")
        else:
            expand_shape = None

        img = self.model_transforms(img.permute(0, 3, 1, 2) / 255.0)
        ret = self.net(img)

        if self.net.classifier_feature == "reshape_embedding":
            # Flatten the spatial tokens since the PPO storage only stores flat vectors.
            ret = rearrange(ret, "b d w h -> b (d w h)")
        else:
            ret = rearrange(ret, "b d -> b 1 d")
        assert ret.shape[1:] == self.output_shape

        # Re-expand the sequence dimension if it is there.
        if expand_shape is not None:
            ret = rearrange(
                ret, "(b t) d -> b t d", b=expand_shape[0], t=expand_shape[1]
            )

        return ret.to(torch.float32)

    @property
    def output_shape(self):
        if self.net.classifier_feature == "reshape_embedding":
            return (np.prod(self.net.patch_embed.grid_size), self.embd_size)
        else:
            return (1, self.embd_size)