from torch import nn


class Decoder(nn.Module):
    """
        Аргументы:
            in_channels (int): кол-во входных каналов
            out_channels (int): кол-во выходных каналов
            kernel_size (int), stride (int): параметры для свертки
        Возвращает:
            объект типа Decoder
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.decoder = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride)

    """
        Декодирует входной tensor x после прохода через Conformer
    """

    def forward(self, x):
        return self.decoder(x)
