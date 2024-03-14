from einops import rearrange
import torch
import torch.nn as nn
from flamingo_pytorch import PerceiverResampler
from .base import VisBridge


class ResamplerVisBridge(VisBridge):
    def __init__(
        self,
        vis_encoder_net,
        output_dim,
        resampler_depth,
        resampler_dim_head,
        resampler_heads,
        num_output_latents,
        use_b16: bool,
        **kwargs,
    ):
        super().__init__()
        if use_b16:
            self._create_type = torch.bfloat16
        else:
            self._create_type = torch.float32

        # NOTE: Make sure vis_encoder_net containers `output_shape`
        num_visual_tokens, vis_token_dim = vis_encoder_net.output_shape

        self.token_resampler = PerceiverResampler(
            dim=vis_token_dim,
            num_media_embeds=num_visual_tokens,
            num_latents=num_output_latents,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_heads,
        )
        self.up_proj = nn.Linear(vis_token_dim, output_dim)
        self.token_resampler.to(self._create_type)
        self._num_output_tokens = num_output_latents

    def forward(self, vis_features):
        """
        Always returns float32 data type regardless of net internal type.
        """

        if len(vis_features.shape) == 4:
            orig_batch_size = vis_features.shape[0]
            vis_features = rearrange(vis_features, "b n t d -> (b n) t d")
        else:
            orig_batch_size = None

        vis_features = vis_features.to(self._create_type)
        embeds = self.token_resampler(vis_features)
        # The token resampler outputs another dimension for some reason...
        embeds = rearrange(embeds, "b 1 t d -> b t d")
        embeds = self.up_proj(embeds)

        if orig_batch_size is not None:
            embeds = rearrange(embeds, "(b n) t d -> b n t d", b=orig_batch_size)
        return embeds.to(torch.float32)

    @property
    def num_tokens(self) -> int:
        return self._num_output_tokens
