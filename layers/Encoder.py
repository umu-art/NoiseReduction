from torch import nn

import Config


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.encoder = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        return x
