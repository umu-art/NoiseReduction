from torch import nn


class DepthwiseConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, bias=True):
        super().__init__()
        assert channels_out % channels_in == 0, 'channels_in must be a divisor of channels_out'
        assert (kernel_size % 2 != 0)
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding='same', groups=channels_in, bias=bias)

    def forward(self, x):
        return self.conv(x)
