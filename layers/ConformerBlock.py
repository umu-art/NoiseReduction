from torch import nn

from layers.FeedForwardModule import FeedForwardModule
from layers.attention import MultiHeadedSelfAttentionModule
from layers.ConvModule import ConvModule
from layers.ResCon import ResCon


class ConformerBlock(nn.Module):
    def __init__(self, size: int, num_heads: int = 4):
        super().__init__()
        self.res_feed_forward_first = ResCon(1, 0.5, FeedForwardModule(size))
        self.res_multi_head = ResCon(1, 1, MultiHeadedSelfAttentionModule(size, num_heads=num_heads))
        self.res_conv = ResCon(1, 1, ConvModule())  # TODO: implement ConvolutionModule
        self.res_feed_forward_second = ResCon(1, 0.5, FeedForwardModule(size))
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        x = self.res_feed_forward_first(x)
        x = self.res_multi_head(x)
        x = self.res_conv(x)
        x = self.res_feed_forward_second(x)
        x = self.layer_norm(x)
        return x
