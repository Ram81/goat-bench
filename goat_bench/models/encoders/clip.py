import clip
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import Normalize

from goat_bench.models.encoders.base import Encoder


class CLIPEncoder(Encoder):
    def __init__(self, name: str = "RN50", device: str = "cuda"):
        super().__init__()
        model, preprocess = clip.load("RN50", device=device)

        self.device = device
        self.model = model
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

    def embed_vision(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        if len(observations.shape) != 4:
            observations = observations.unsqueeze(0)
        observations = observations.permute(0, 3, 1, 2)
        transformed_img = self.model_transforms(observations).to(self.device)
        with torch.inference_mode():
            embedding = self.model.encode_image(transformed_img)
        return embedding.squeeze().detach().cpu().numpy()

    def embed_language(self, observations):
        raise NotImplementedError

    @property
    def perception_embedding_size(self) -> int:
        return 1024

    @property
    def language_embedding_size(self) -> int:
        raise NotImplementedError
