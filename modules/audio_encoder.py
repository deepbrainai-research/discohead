import torch
import torch.nn as nn

from .layers import conv1d, lstm, linear


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            conv1d(192, 512, 3, 1, 1), nn.ReLU(),
            conv1d(512, 512, 3, 1, 1), nn.ReLU(),
            conv1d(512, 512, 3, 1, 1), nn.ReLU(),
            conv1d(512, 512, 3, 1, 1), nn.ReLU())

        self.rnn = lstm(512, 512)

        self.decoder = nn.Sequential(
            linear(512, 256), nn.ReLU(),
            linear(256, 256))
        

    def forward(self, x):
        x = self.encoder(x) # (n, c, l)
        x, _ = self.rnn(x.permute(0, 2, 1)) # (n, l, c)
        x = self.decoder(x[:,-1]) # (n, c)
        return x

