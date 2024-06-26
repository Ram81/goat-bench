import torch
from lavis.models import load_model_and_preprocess
from torchvision import transforms

from goat_bench.models.encoders.base import Encoder


class Blip2Encoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = kwargs["device"]

        model, vis_processors, _ = load_model_and_preprocess(
            name=kwargs["name"],
            model_type=kwargs["model_type"],
            is_eval=True,
            device=self.device,
        )
        self.model = model.to(self.device)
        self.model_transforms = vis_processors["eval"]

    def embed_vision(self, observations):
        raise NotImplementedError

    def embed_language(self, observations):
        raise NotImplementedError

    @property
    def perception_embedding_size(self) -> int:
        raise NotImplementedError

    @property
    def language_embedding_size(self) -> int:
        raise NotImplementedError

