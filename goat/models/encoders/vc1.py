import numpy as np
import torch
from torchvision import transforms
from vc_models.models.vit import model_utils

from goat.models.encoders.base import Encoder


class VC1Encoder(Encoder):
    def __init__(
        self, name: str = model_utils.VC1_LARGE_NAME, device: str = "cuda"
    ):
        super().__init__()
        model, _, model_transforms, _ = model_utils.load_model(name)

        self.device = device
        self.model = model.to(self.device)
        self.model_transforms = model_transforms

    def embed_vision(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        if len(observations.shape) != 4:
            observations = observations.unsqueeze(0)
        observations = observations.permute(0, 3, 1, 2)
        transformed_img = self.model_transforms(observations).to(self.device)
        with torch.inference_mode():
            embedding = self.model(transformed_img)
        return embedding.squeeze().detach().cpu().numpy()

    def embed_language(self, observations):
        raise NotImplementedError

    @property
    def perception_embedding_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def language_embedding_size(self) -> int:
        raise NotImplementedError
