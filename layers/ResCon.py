from torch import nn


class ResCon(nn.Module):
    """
        Класс-обертка для слоя
        Если слой принимает x и возвращает y, то в ResCon возвращается ax + by
        Аргументы:
            a (float): коэффициент перед входом
            b (float): коэффициент перед выходом
            module (nn.Module): слой для обертки
    """

    def __init__(self, a: float, b: float, module: nn.Module):
        super().__init__()
        self.a = a
        self.b = b
        self.module = module

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, измененный по правилам выше
    """

    def forward(self, x):
        return self.a * x + self.b * self.module(x)
