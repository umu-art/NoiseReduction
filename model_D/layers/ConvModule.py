from torch import nn

from model_D.activators.GLU import GLU
from model_D.activators.Swish import Swish
from model_D.layers.DepthwiseConv import DepthwiseConv
from model_D.layers.PointwiseConv import PointwiseConv
from model_D.layers.GlobalLayerNorm import GlobalLayerNorm


class ConvModule(nn.Module):
    """
        Аргументы:
            in_channels (int): количество входных каналов
            out_channels (int): количество выходных каналов
            dropout_p (float): вероятность dropout'a
        Выход:
            объект типа ConvModule
    """

    def __init__(self, in_channels: int, kernel_size: int, dropout_p: float):
        super().__init__()
        self.layer_norm = GlobalLayerNorm(in_channels)
        self.pointwise_conv_first = PointwiseConv(in_channels, in_channels * 2)
        self.glu = GLU(1)
        self.depthwise_conv = DepthwiseConv(in_channels, in_channels, kernel_size)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish = Swish()
        self.pointwise_conv_second = PointwiseConv(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_p)

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.pointwise_conv_first(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv_second(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x
