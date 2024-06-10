# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for GPU >= Ampere and PyTorch >= 1.12
from functools import partial

from .croco.models.blocks import DecoderBlock, PatchEmbed
from .croco.models.pos_embed import get_2d_sincos_pos_embed, RoPE2D

class Croco2ViTDecoder(nn.Module):
    """
    Croco2ViTDecoder: A decoder module for the CroCo model.

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
                 norm_im2_in_dec=True, pos_embed='cosine',
                 adapter=False, adapter_bottleneck=64, adapter_scalar='0.1', adapter_style='parallel'):
        
        super(Croco2ViTDecoder, self).__init__()

        # Patch embeddings (with initialization done as in MAE)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

        self.pos_embed = pos_embed
        if pos_embed == 'cosine':
            # Positional embedding of the encoder
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # Position embedding in each block
            self.rope = None  # Nothing for cosine
        elif pos_embed.startswith('RoPE'):  # e.g., RoPE100
            self.dec_pos_embed = None  # Nothing to add in the decoder with RoPE
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed ' + pos_embed)

        # Decoder
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, adapter, adapter_bottleneck, adapter_scalar, adapter_style)

        # Initialize weights
        self.initialize_weights()

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, adapter, adapter_bottleneck, adapter_scalar, adapter_style):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # Transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # Transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope,
                         adapter=adapter, adapter_bottleneck=adapter_bottleneck, adapter_scalar=adapter_scalar, adapter_style=adapter_style)
            for i in range(dec_depth)])
        # Final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # linears and layer norms
        # self.apply(self._init_weights)
        self.recursive_apply(self)
        
    def recursive_apply(self, m):
        for name, m2 in m.named_children():
            if "adaptmlp" in name:
                return
            else:
                self.recursive_apply(m2)
        self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _decoder(self, feat1, pos1, feat2, pos2, return_all_blocks=False):
        """
        Decoder function for Croco2ViTDecoder.

        Args:
            feat1: Features from the first image.
            pos1: Position information for the first image.
            feat2: Features from the second image.
            pos2: Position information for the second image.
            return_all_blocks: If True, return the features at the end of every block instead of just the features from the last block.

        Returns:
            Output features from the decoder.
        """
        # Encoder to decoder layer
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        f1_ = visf1
        # Add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # Apply Transformer blocks
        out = f1_
        out2 = f2
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)
        return out

    def forward(self, feat1, pos1, feat2, pos2):
        """
        Forward pass for Croco2ViTDecoder.

        Args:
            feat1: Encoder features from the first image.
            pos1: Position information for the first image.
            feat2: Features from the second image.
            pos2: Position information for the second image.

        Returns:
            Decoded features.
        """
        # Decoder
        decfeat = self._decoder(feat1, pos1, feat2, pos2)
        return decfeat
