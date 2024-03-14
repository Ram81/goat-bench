import torch
import numpy as np
import gym.spaces as spaces
from goat.models.visual_encoders.clip_encoder import ClipEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define a random observation dictionary with tensors for images
observation_dict = {
    "image1": torch.randn(1, 224, 224, 3).to(device),  # Example image tensor
    # "image2": torch.randn(1, 224, 224, 3)   # Example image tensor
}

# NOTE: CLIP does not work with more than one image! 3 channels only.
# Define the observation space
obs_space = spaces.Dict({
    "image1": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
    # "image2": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
})

# Test with linear proj (cls token)
# Create an instance of ClipEncoder
clip_encoder = ClipEncoder(
    im_obs_space=obs_space,
    use_b16=False,
    classifier_feature="use_cls_token",
    checkpoint_name="ViT-B/32"
)

# Pass the observation dictionary through the encoder
output = clip_encoder.forward(observation_dict)
# Check the output shape
print("Output shape:", output.shape)
print("Encoder output shape: ", clip_encoder.output_shape)


## Test without linear proj
clip_encoder = ClipEncoder(
    im_obs_space=obs_space,
    use_b16=False,
    classifier_feature="reshape_embedding",
    checkpoint_name="ViT-B/32",
    linear_proj=False,
)
# Pass the observation dictionary through the encoder
output = clip_encoder.forward(observation_dict)
# Check the output shape
print("Output shape:", output.shape)
print("Encoder output shape: ", clip_encoder.output_shape)