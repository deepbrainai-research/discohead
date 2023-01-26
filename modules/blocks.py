import torch.nn as nn
import torch.nn.functional as F

from .layers import linear, conv2d, conv3d, mod_conv2d


activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}

class conv2d_bn_relu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation='relu'
    ):
        super(conv2d_bn_relu, self).__init__()
        self.conv = conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.act = activations[activation]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class down_block2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_block2d, self).__init__()
        self.conv = conv2d_bn_relu(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        return self.pool(self.conv(x))


class res_block2d(nn.Module):
    def __init__(self, in_channels):
        super(res_block2d, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(),
            conv2d(in_channels, in_channels, 3, 1, 1))

    def forward(self, x):
        return x + self.block(x)


class res_mod_block2d(nn.Module):
    def __init__(self, in_channels, in_features):
        super(res_mod_block2d, self).__init__()
        self.block = nn.ModuleList([
            mod_conv2d(in_channels, in_channels, 3, 1, in_features),
            mod_conv2d(in_channels, in_channels, 3, 1, in_features)])

    def forward(self, x, y):
        r = x
        for i in range(len(self.block)):
            r = self.block[i](r, y)
        return x + r


class up_block2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_block2d, self).__init__()
        self.conv = conv2d_bn_relu(in_channels, out_channels)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2))

