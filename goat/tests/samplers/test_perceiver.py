import torch
import numpy as np
import gym.spaces as spaces
from goat.models.visual_encoders.clip_encoder import ClipEncoder
from goat.models.samplers.perceiver import ResamplerVisBridge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_space = spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
})

clip_encoder = ClipEncoder(
    im_obs_space=obs_space,
    use_b16=False,
    classifier_feature="use_cls_token",
    checkpoint_name="ViT-B/32",
    linear_proj=False
)


# Define Perceiver Resampler-based Model
vis_encoder_net = clip_encoder
output_dim = 512
resampler_depth = 1
resampler_dim_head = 64
resampler_heads = 4
num_output_latents = 10
use_b16 = False  # Example flag for using bfloat16

resampler_model = ResamplerVisBridge(
    vis_encoder_net=vis_encoder_net,
    output_dim=output_dim,
    resampler_depth=resampler_depth,
    resampler_dim_head=resampler_dim_head,
    resampler_heads=resampler_heads,
    num_output_latents=num_output_latents,
    use_b16=use_b16
)
resampler_model = resampler_model.to(device)

# Generate random input tensor for CLIP encoder
batch_size = 2
image_height, image_width = 224, 224
random_input = torch.randn(batch_size, image_height, image_width, 3).to(device)

# Pass through CLIP encoder
clip_output = clip_encoder({"image": random_input})

# Pass through Perceiver Resampler-based model
resampler_output = resampler_model(clip_output)

print("CLIP Encoder Output Shape:", clip_output.shape)
print("Perceiver Resampler Output Shape:", resampler_output.shape)