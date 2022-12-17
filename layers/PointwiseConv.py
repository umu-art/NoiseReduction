from torch import nn


class PointwiseConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, bias=True):
        super().__init__()

        self.conv = nn.Conv1d(channels_in, channels_out, 1, bias=bias)

    def forward(self, x):
        return self.conv(x)
