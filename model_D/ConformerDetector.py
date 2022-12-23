import torch
from torch import nn

import model_D.Config
from model_D.layers.Encoder import Encoder
from model_D.layers.Conformer import Conformer


class ConformerDetector(nn.Module):
    def __init__(self, size: int, conf_blocks_num: int,
                 conv_kernel_size: int, in_channels: int, kernel_size: int, stride: int,
                 num_heads: int = 4, expansion_factor: int = 4,
                 dropout_conv: float = 0.1, dropout_multi_head: float = 0.1, dropout_feed_forward: float = 0.1):
        super().__init__()
        self.encoder = Encoder(in_channels, size, kernel_size, stride)
        self.conformer = Conformer(size, conf_blocks_num, conv_kernel_size,
                                   num_heads, expansion_factor, dropout_conv, dropout_multi_head, dropout_feed_forward)
        self.lin_first = nn.Linear(size, size)
        self.activator = nn.PReLU()
        self.lin_second = nn.Linear(size, 1)

    """
        Аргументы:
            x (toch.Tensor): (b, T) -> вход
        Возвращает:
            out (torch.Tensor): (b) -> характеристика для каждого из выборки
    """

    def forward(self, x: torch.Tensor):  # (b, T)
        x = torch.unsqueeze(x, 1)  # (b, 1, T)
        encoded = self.encoder(x)  # (b, size, T)
        encoded = torch.permute(encoded, [0, 2, 1])  # (b, T, size)
        conformed = self.conformer(encoded)  # (b, T, size)
        x = torch.mean(conformed, dim=1)  # (b, size)
        x = self.lin_first(x)  # (b, size)
        x = self.activator(x)  # (b, size)
        x = self.lin_second(x)  # logits (b, 1)
        x = torch.squeeze(x, 1)  # (b)
        return x


if __name__ == '__main__':  # test
    mt = torch.rand([Config.batch_size, 100])
    conf = ConformerDetector(Config.size, Config.conf_blocks_num, Config.conv_kernel_size, 1, Config.w_len,
                             Config.w_len // 2)
    mt = conf(mt)
    print(mt.shape)
