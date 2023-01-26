import torch
import torch.nn as nn
import torch.nn.functional as F


class linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        # nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # initialize
        # kernel: glorot_uniform
        # recurrent: orthogonal
        # bias: zeros
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                for i in range(4):
                    o = param.shape[0] // 4
                    nn.init.xavier_uniform_(param[i*o:(i+1)*o])
            elif 'weight_hh' in name:
                for i in range(4):
                    o = param.shape[0] // 4
                    nn.init.orthogonal_(param[i*o:(i+1)*o])
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        return self.lstm(x)


class mod_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, in_features):
        super(mod_conv2d, self).__init__()

        self.mod = linear(in_features, in_channels)
        nn.init.zeros_(self.mod.linear.weight)
        nn.init.constant_(self.mod.linear.bias, 1.0)

        self.bn = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU()

        weight = torch.zeros(
            (1, out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.weight = nn.Parameter(weight)
        nn.init.xavier_uniform_(self.weight)

        self.padding = padding


    def forward(self, x, y):
        x = self.bn(x)
        x = self.relu(x)

        n, _, h, w = x.shape
        _, o, i, k, k = self.weight.shape

        # Modulate
        scale = self.mod(y).view(n, 1, i, 1, 1)
        weight = self.weight * scale # (n, o, i, k, k)

        # Demodulate
        demod = weight.pow(2).sum([2, 3, 4], keepdim=True)
        demod = torch.rsqrt(demod + 1e-8) # (n, o, 1, 1, 1)
        weight = weight * demod # (n, o, i, k, k)

        x = x.view(1, n * i, h, w)
        weight = weight.view(n * o, i, k, k)
        x = F.conv2d(x, weight, bias=None, padding=self.padding, groups=n)
        x = x.view(n, o, h, w)

        return x
