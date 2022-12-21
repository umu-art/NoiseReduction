from torch import nn


class Encoder(nn.Module):
    """
        Аргументы:
            in_channels (int): кол-во входных каналов
            out_channels (int): кол-во выходных каналов
            kernel_size (int), stride (int): параметры для свертки
        Возвращает:
            объект типа Encoder
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.encoder = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride)
        self.relu = nn.ReLU()

    """
        Кодирует входной tensor x перед проходом через Conformer
    """

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        return x
