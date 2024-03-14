from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from .base import VisBridge


class MlpVisBridge(VisBridge):
    """
    For operating over single token inputs.
    """

    def __init__(
        self,
        vis_encoder_net,
        llm,
        state_obs_space,
        hidden_size,
        cfg,
        **kwargs,
    ):
        super().__init__()
        llm_input_size = llm.d_model
        if not hasattr(vis_encoder_net, "embd_size"):
            if hasattr(vis_encoder_net, "output_shape"):
                input_size = np.prod(vis_encoder_net.output_shape)
            else:
                raise ValueError("Visual encoder must specify output size.")
        else:
            input_size = vis_encoder_net.embd_size

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.ReLU(True),
        )
        self._state_obs_space = state_obs_space
        visual_dim = hidden_size
        input_dim = visual_dim + sum(
            space.shape[0] for space in state_obs_space.spaces.values()
        )
        self.state_token_proj = nn.Linear(input_dim, llm_input_size)

    def forward(self, vis_features, obs):
        # There is only 1 visual token. Extract this token and expand.

        if len(vis_features.shape) == 4:
            # Operate on the only visual token.
            assert vis_features.shape[2] == 1

            batch_size = vis_features.shape[0]
            # Flatten and remove #token dim.
            vis_features = rearrange(vis_features, "b r 1 d -> (b r) d")
            vis_features = self.visual_fc(vis_features)
            vis_features = rearrange(vis_features, "(b r) d -> b r d", b=batch_size)
        else:
            assert vis_features.shape[1] == 1
            vis_features = vis_features[:, 0]

            vis_features = self.visual_fc(vis_features)

        state_features = [obs[k] for k in self._state_obs_space.keys()]

        if vis_features is None:
            hidden_window = torch.cat(state_features, dim=-1)
        elif len(state_features) == 0:
            hidden_window = vis_features
        else:
            hidden_window = torch.cat([vis_features, *state_features], dim=-1)

        hidden_window = self.state_token_proj(hidden_window)
        return hidden_window.unsqueeze(-2)

    @property
    def num_tokens(self) -> int:
        return 1