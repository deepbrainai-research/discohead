import torch
import torch.nn as nn

from .layers import linear
from .utils import AntiAliasInterpolation2d
from .blocks import down_block2d


class EyesEncoder(nn.Module):
    def __init__(self):
        super(EyesEncoder, self).__init__()
        self.down = AntiAliasInterpolation2d(3, 0.5) # 1/2 resolution

        self.encoder = nn.Sequential(
            down_block2d(3, 32),
            down_block2d(32, 64),
            down_block2d(64, 64),
            down_block2d(64, 64),
            down_block2d(64, 64),
            nn.AvgPool2d(kernel_size=(4, 4)))

        self.predictor = nn.Sequential(
            linear(64, 32), nn.ReLU(),
            linear(32, 8))
        

    def forward(self, x):
        # x: (n, c, 32, 64)
        x = self.down(x)
        x = self.encoder(x)
        x = self.predictor(x[...,0,0])
        return x
