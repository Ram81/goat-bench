import numpy as np
import torch

from goat.models.encoders.base import Encoder
from goat.models.encoders.croco2_vit_encoder import Croco2ViTEncoder
from torchvision.transforms import Normalize, Compose, Resize, InterpolationMode, CenterCrop

class Croco2Encoder(Encoder):
    def __init__(
        self, ckpt_path: str = 'goat/models/encoders/croco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', device: str = "cuda"
    ):
        super().__init__()
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        model_transforms = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        self.device = device
        ckpt = torch.load(ckpt_path, 'cpu')
        self.model = Croco2ViTEncoder(**ckpt.get('croco_kwargs', {})).to(device)
        self.model_transforms = model_transforms

    def embed_vision(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        if len(observations.shape) != 4:
            observations = observations.unsqueeze(0)
        observations = observations.permute(0, 3, 1, 2)
        transformed_img = self.model_transforms(observations).to(self.device)
        with torch.inference_mode():
            feat, pos = self.model(transformed_img)
        return feat.squeeze().detach().cpu().numpy(), pos.squeeze().detach().cpu().numpy()

    def embed_language(self, observations):
        raise NotImplementedError

    @property
    def perception_embedding_size(self) -> int:
        return self.model.enc_embed_dim

    @property
    def language_embedding_size(self) -> int:
        raise NotImplementedError
