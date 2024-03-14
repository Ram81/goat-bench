# Reference: https://github.com/apple/ml-llarp/blob/main/llarp/policies/visual_encoders.py
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc
import torch.nn as nn

class VisualEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, obs):
        """
        Must return shape [batch_size, # visual embeds, token embed dim]
        """

    @property
    @abc.abstractmethod
    def output_shape(self):
        pass