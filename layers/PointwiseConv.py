from torch import nn


class PointwiseConv(nn.Module):
    """
        PointwiseConv - свертка, которая использует ядро размера 1.
        Аргументы:
            channels_in (int) - кол-во входных каналов
            channels_out (int) - кол-во выходных каналов
            bias (bool) - использовать/не использовать сдвиг
        Вход:
            x (tensor) - тензор, содержащий входной вектор
        Выход:
            **output** (tensor) -  тензор после применения свертки
    """

    def __init__(self, channels_in: int, channels_out: int, bias=True):
        super().__init__()

        self.conv = nn.Conv1d(channels_in, channels_out, 1, bias=bias)

    def forward(self, x):
        return self.conv(x)
