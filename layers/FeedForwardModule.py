from torch import nn

from activators.Swish import Swish

from layers.GlobalLayerNorm import GlobalLayerNorm


class FeedForwardModule(nn.Module):
    """
        FeedForwardModule - просто объединяет все слои.
        Аргументы:
            size (int) - размер входных данных
            dropout_p (float) - вероятность обнуления элемента
            expansion_factor (int) - коэффициент количества доп. каналов
        Вход:
            x (tensor) - входной тензор
        Выход:
            **output** - тензор после применения всех слоёв
    """

    def __init__(self, size: int, dropout_p: float, expansion_factor: int):
        super().__init__()
        self.layer_norm = GlobalLayerNorm(size)
        self.lin_first = nn.Linear(size, size * expansion_factor)
        self.swish = Swish()
        self.dropout_first = nn.Dropout(p=dropout_p)
        self.lin_second = nn.Linear(size * expansion_factor, size)
        self.dropout_second = nn.Dropout(p=dropout_p)

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.lin_first(x)
        x = self.swish(x)
        x = self.dropout_first(x)
        x = self.lin_second(x)
        x = self.dropout_second(x)
        return x
