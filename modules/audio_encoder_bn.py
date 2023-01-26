import torch
import torch.nn as nn

from .layers import conv1d, conv2d, linear


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            conv1d(192, 256, 7, 1, 0), nn.BatchNorm1d(256), nn.ReLU(),
            conv1d(256, 256, 5, 1, 0), nn.BatchNorm1d(256), nn.ReLU(),
            conv1d(256, 256, 5, 1, 0), nn.BatchNorm1d(256), nn.ReLU(),
            conv1d(256, 256, 5, 1, 0), nn.BatchNorm1d(256), nn.ReLU(),
            conv1d(256, 256, 5, 1, 0), nn.BatchNorm1d(256), nn.ReLU())

        self.predictor = nn.Sequential(
            linear(256*9, 256), nn.ReLU(),
            linear(256, 256), nn.ReLU(),
            linear(256, 256), nn.ReLU(),
            linear(256, 256), nn.ReLU(),
            linear(256, 256), nn.ReLU(),
            linear(256, 256))
        

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.predictor(x)
        return x
