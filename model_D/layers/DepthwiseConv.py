from torch import nn


class DepthwiseConv(nn.Module):
    """
            DepthWiseConv - свертка.
            Аргументы:
                channels_in (int) - кол-во входных каналов
                channels_out (int) - кол-во выходных каналов
                kernel_size (int) - размер свертки
                bias (bool) - использовать/не использовать сдвиг
            Вход:
                x (tensor) - тензор, содержащий входной вектор
            Выход:
                **output** (tensor) -  тензор после применения свертки
    """
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, bias=True):
        super().__init__()
        assert channels_out % channels_in == 0, 'channels_in must be a divisor of channels_out'
        assert (kernel_size % 2 != 0)
        self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding='same', groups=channels_in, bias=bias)

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        return self.conv(x)
