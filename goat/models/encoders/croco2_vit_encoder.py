# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # For GPU >= Ampere and PyTorch >= 1.12
from functools import partial

from .croco.models.blocks import Block, PatchEmbed
from .croco.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D

class Croco2ViTEncoder(nn.Module):
    """
    Croco2ViTEncoder: An encoder module for the CroCo model.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        mask_ratio (float): Ratio of masked tokens.
        enc_embed_dim (int): Encoder feature dimension.
        enc_depth (int): Encoder depth.
        enc_num_heads (int): Number of heads in the encoder's transformer block.
        dec_embed_dim (int): Decoder feature dimension.
        dec_depth (int): Decoder depth.
        dec_num_heads (int): Number of heads in the decoder's transformer block.
        mlp_ratio (int): Ratio for MLP layers.
        norm_layer (callable): Normalization layer constructor.
        norm_im2_in_dec (bool): Whether to apply normalization to the 'memory' (second image) in the decoder.
        pos_embed (str): Type of positional embedding (either 'cosine' or 'RoPE100').
    """

    def __init__(self, img_size=224, patch_size=16, mask_ratio=0.9, enc_embed_dim=768,
                 enc_depth=12, enc_num_heads=12, dec_embed_dim=512, dec_depth=8,
                 dec_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True, pos_embed='cosine'):
        super(Croco2ViTEncoder, self).__init__()

        # Patch embeddings (with initialization done as in MAE)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

        self.pos_embed = pos_embed
        if pos_embed == 'cosine':
            # Positional embedding of the encoder
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(self.patch_embed.num_patches**0.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            self.rope = None  # Nothing for cosine
        elif pos_embed.startswith('RoPE'):  # e.g., RoPE100
            self.enc_pos_embed = None  # Nothing to add in the encoder with RoPE
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed ' + pos_embed)

        # Transformer for the encoder
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Patch embed
        self.patch_embed._init_weights()
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_image(self, image, return_all_blocks=False):
        """
        Encode the image into patches and apply masking if needed.

        Args:
            image (Tensor): Input image with shape B x 3 x img_size x img_size.
            return_all_blocks (bool): If True, return features at the end of every block instead of the last block.

        Returns:
            Encoded features, positions, and masks (if masking is performed).
        """
        # Embed the image into patches (x has size B x Npatches x C)
        # and get the position of each returned patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)
        # Add positional embedding without cls token
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]
        B, N, C = x.size()
        posvis = pos
        # Apply the transformer encoder and normalization
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos

    def forward(self, img1):
        """
        Forward pass for Croco2ViTEncoder.

        Args:
            img1 (Tensor): Input image with shape B x 3 x img_size x img_size.

        Returns:
            Encoded features and positions.
        """
        feat1, pos1 = self._encode_image(img1)
        return feat1, pos1
