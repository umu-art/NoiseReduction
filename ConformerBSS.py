import torch
from torch import nn

import Config
from layers.Encoder import Encoder
from layers.Decoder import Decoder
from layers.Conformer import Conformer


class ConformerBSS(nn.Module):
    def __init__(self, size: int, conf_blocks_num: int,
                 conv_kernel_size: int, in_channels: int, kernel_size: int, stride: int,
                 num_heads: int = 4, expansion_factor: int = 4,
                 dropout_conv: float = 0.1, dropout_multi_head: float = 0.1, dropout_feed_forward: float = 0.1):
        super().__init__()
        self.encoder = Encoder(in_channels, size, kernel_size, stride)
        self.conformer = Conformer(size, conf_blocks_num, conv_kernel_size,
                                   num_heads, expansion_factor, dropout_conv, dropout_multi_head, dropout_feed_forward)
        self.sigmoid = nn.Sigmoid()
        self.decoder = Decoder(size, in_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor):  # (b, T)
        x = torch.unsqueeze(x, 1)  # (b, 1, T)
        encoded = self.encoder(x)  # (b, size, T)
        encoded = torch.permute(encoded, [0, 2, 1])  # (b, T, size)
        conformed = self.conformer(encoded)  # (b, T, size * 2)
        conformed = torch.reshape(conformed, [conformed.shape[0], conformed.shape[1], conformed.shape[2] // 2,
                                              2])  # (b, T, size, 2)
        mask = self.sigmoid(conformed)  # (b, T, size, 2)
        encoded = encoded.unsqueeze(-1)  # (b, T, size, 1)
        x = encoded * mask  # (b, T, size, 2)
        x = torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])  # (b, T, size * 2)
        x = x.permute(0, 2, 1)  # (b, size * 2, T)
        x = torch.reshape(x, [x.shape[0], 2, x.shape[1] // 2, x.shape[2]])  # (b, 2, size, T)
        x = torch.reshape(x, [x.shape[0] * x.shape[1], x.shape[2], x.shape[3]])  # (2 * b, size, T)
        x = self.decoder(x)  # (2 * b, 1, T)
        x = torch.squeeze(x, 1)  # (2 * b, T)
        x = torch.reshape(x, [x.shape[0] // 2, 2, x.shape[1]])  # (b, 2, T)
        return x


if __name__ == '__main__':  # test
    mt = torch.rand([Config.batch_size, 1000])
    conf = ConformerBSS(Config.size, Config.conf_blocks_num, Config.conv_kernel_size, 1, Config.w_len,
                        Config.w_len // 2)
    mt = conf(mt)
    print(mt.shape)
