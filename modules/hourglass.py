import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import down_block2d, up_block2d


class Encoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            if i == 0:
                in_channels = in_features 
            else:
                in_channels = min(max_features, block_expansion * (2 ** i))
            out_channels = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(down_block2d(in_channels, out_channels))

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_channels = min(max_features, block_expansion * (2 ** (i + 1)))
            if i != num_blocks - 1:
                in_channels = 2 * in_channels
            out_channels = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(up_block2d(in_channels, out_channels))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    def __init__(
        self,
        block_expansion=32,
        in_features=3,
        num_blocks=5,
        max_features=1024
    ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))
