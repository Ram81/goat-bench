# Reference: https://github.com/apple/ml-llarp/blob/main/llarp/policies/vis_bridge.py
import abc
import torch.nn as nn


class VisBridge(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, vis_features, obs):
        pass

    @property
    @abc.abstractmethod
    def num_tokens(self) -> int:
        pass