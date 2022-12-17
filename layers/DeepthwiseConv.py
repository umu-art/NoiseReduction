from torch import nn


class DeepthwiseConv:
    def __init__(self, channels_in, channels_out, kernel_size):
        super().__init__()
        assert channels_out % channels_in == 0, 'channels_in must be a divisor of channels_out'
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding='same', groups=channels_in)

    def forward(self, x):
        return self.conv(x)
