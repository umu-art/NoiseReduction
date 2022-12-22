from torch import nn

from model_NR.layers.FeedForwardModule import FeedForwardModule
from model_NR.layers.AttentionLayer import MultiHeadedSelfAttentionModule
from model_NR.layers.ConvModule import ConvModule
from model_NR.layers.ResCon import ResCon


class ConformerBlock(nn.Module):
    """
    Аргументы:
        size (int): размер входных данных
        conv_kernel_size (int): размер свертки, используемый в слое ConvModule.py
        num_heads (int): Количество heads в AttentionLayer.py
        expansion_factor (int): множитель в линейном слое
        dropout_conv (float): вероятность dropout'а в слое ConvModule.py
        dropout_feed_forward (int): вероятность dropout'а в слое FeedForwardModule.py
        dropout_multi_head (float): вероятность dropout'а в слое MultiHeadedSelfAttentionModule.py
    Возвращает:
        Объект типа ConformerBlock
    """

    def __init__(self, size: int, conv_kernel_size: int, num_heads: int,
                 expansion_factor: int,
                 dropout_conv: float, dropout_feed_forward: float, dropout_multi_head: float):
        super().__init__()
        self.res_feed_forward_first = ResCon(1, 0.5, FeedForwardModule(size=size, dropout_p=dropout_feed_forward,
                                                                       expansion_factor=expansion_factor))
        self.res_multi_head = ResCon(1, 1, MultiHeadedSelfAttentionModule(d_model=size, num_heads=num_heads,
                                                                          dropout_p=dropout_multi_head))
        self.res_conv = ResCon(1, 1, ConvModule(in_channels=size, kernel_size=conv_kernel_size,
                                                dropout_p=dropout_conv))
        self.res_feed_forward_second = ResCon(1, 0.5, FeedForwardModule(size, dropout_p=dropout_feed_forward,
                                                                        expansion_factor=expansion_factor))
        self.layer_norm = nn.LayerNorm(size)

    """
    Аргументы: 
        x (torch.tensor): входные данные
    Возвращает:
        out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        x = self.res_feed_forward_first(x)
        x = self.res_multi_head(x)
        x = self.res_conv(x)
        x = self.res_feed_forward_second(x)
        x = self.layer_norm(x)
        return x
