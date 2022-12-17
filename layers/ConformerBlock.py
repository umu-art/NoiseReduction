from torch import nn

from layers.FeedForwardModule import FeedForwardModule
from layers.attention import MultiHeadedSelfAttentionModule
from layers.ConvModule import ConvModule


class ConformerBlock(nn.Module):
    def __init__(self, size: int, num_heads: int = 4):
        super().__init__()
        self.feed_forward_first = FeedForwardModule(size)
        self.multi_head = MultiHeadedSelfAttentionModule(size, num_heads=num_heads)
        self.conv = ConvModule()  # TODO: implement ConvolutionModule
        self.feed_forward_second = FeedForwardModule(size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x):
        x_after_first_feed_forward = self.feed_forward_first(x)
        x_before_multi_head = x * x_after_first_feed_forward / 2
        x_after_multi_head = self.multi_head(x_before_multi_head)
        x_before_conv = x_before_multi_head + x_after_multi_head
        x_after_conv = self.conv(x_before_conv)
        x_before_second_feed_forward = x_before_conv + x_after_conv
        x_after_second_feed_forward = self.feed_forward_second(x_before_second_feed_forward)
        x_before_norm = x_before_second_feed_forward + x_after_second_feed_forward / 2
        return self.layer_norm(x_before_norm)
