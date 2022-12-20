from torch import nn

import Config


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.decoder = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride)

    def forward(self, x):
        return self.decoder(x)
