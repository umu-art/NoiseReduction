from torch import nn

from layers.FeedForwardModule import FeedForwardModule
from layers.AttentionLayer import MultiHeadedSelfAttentionModule
from layers.ConvModule import ConvModule
from layers.ResCon import ResCon


class ConformerBlock(nn.Module):
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

    def forward(self, x):
        x = self.res_feed_forward_first(x)
        x = self.res_multi_head(x)
        x = self.res_conv(x)
        x = self.res_feed_forward_second(x)
        x = self.layer_norm(x)
        return x
