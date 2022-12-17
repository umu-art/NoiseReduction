from torch import nn

from layers.StftLayer import StftLayer
from layers.ConformerBlock import ConformerBlock
from layers.IStftLayer import IStftLayer


class Conformer(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, window_: str, size: int, conf_blocks_size: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_ = window_
        self.size = size
        self.conf_blocks_size = conf_blocks_size
        self.f = (n_fft // 2) + 1

        self.stft = StftLayer(n_fft, hop_length, win_length, window_)
        self.lin_first = nn.Linear(self.f, size)
        self.conf_blocks = [ConformerBlock(size) for i in range(conf_blocks_size)]
        self.lin_second = nn.Linear(size, self.f)
        self.i_stft = IStftLayer(n_fft, hop_length, win_length, window_)

    def forward(self, x):
        x = self.stft(x)
        x = self.lin_first(x)
        for block in self.conf_blocks:
            x = block(x)
        x = self.lin_second(x)
        x = self.i_stft(x)
        return x
