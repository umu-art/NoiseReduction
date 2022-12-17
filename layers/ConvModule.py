from torch import nn

from activators.GLU import GLU
from activators.Swish import Swish
from layers.DepthwiseConv import DepthwiseConv
from layers.PointwiseConv import PointwiseConv


class ConvModule(nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_p: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.pointwise_conv_first = PointwiseConv(in_channels, in_channels * 2)
        self.glu = GLU(1)
        self.depthwise_conv = DepthwiseConv(in_channels, in_channels,kernel_size)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish = Swish()
        self.pointwise_conv_second = PointwiseConv(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        start_x = x
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
        return start_x + x

