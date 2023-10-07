import abc

import torch.nn as nn


class Encoder(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def embed_vision(self, observations):
        pass

    @abc.abstractmethod
    def embed_language(self, observations):
        pass

    @property
    @abc.abstractmethod
    def perception_embedding_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def language_embedding_size(self) -> int:
        pass
