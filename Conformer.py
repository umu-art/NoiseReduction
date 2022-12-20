from torch import nn

from layers.StftLayer import StftLayer
from layers.ConformerBlock import ConformerBlock
from layers.IStftLayer import IStftLayer


class Conformer(nn.Module):
    """
        Аргументы:
            n_fft (int): размер свертки (обычно степень двойки)
            hop_length, win_length, window_: параметры для STFT и ISTFT
            size: размер на выходе линейного преобразования
            conf_blocks_num: количество ConformerBlock'ов
            conv_kernel_size: параметр для DepthWise слоя
            num_heads, expansion_factor, dropout_conv, dropout_multi_head, dropout_feed_forward: параметры для ConformerBlock (см. ConformerBlock.py)
        Возвращает:
            Объект типа Conformer (основная Model обучения)
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, window_: str, size: int, conf_blocks_num: int,
                 conv_kernel_size: int, num_heads: int = 4, expansion_factor: int = 4,
                 dropout_conv: float = 0.1, dropout_multi_head: float = 0.1, dropout_feed_forward: float = 0.1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_ = window_
        self.size = size
        self.conf_blocks_size = conf_blocks_num
        self.f = (n_fft // 2) + 1

        self.stft = StftLayer(n_fft, hop_length, win_length, window_)
        self.layer_norm = nn.LayerNorm(self.f, self.f)
        self.lin_first = nn.Linear(self.f, size)
        self.conf_blocks = nn.ModuleList(
            [ConformerBlock(size=size, num_heads=num_heads,
                            conv_kernel_size=conv_kernel_size, expansion_factor=expansion_factor,
                            dropout_conv=dropout_conv, dropout_multi_head=dropout_multi_head,
                            dropout_feed_forward=dropout_feed_forward)
             for _ in
             range(conf_blocks_num)])
        self.lin_second = nn.Linear(size, self.f)
        self.i_stft = IStftLayer(n_fft, hop_length, win_length, window_)

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, x):
        length = x.shape[-1]
        spec, mag = self.stft(x)
        x = self.layer_norm(mag)
        x = self.lin_first(x)
        for block in self.conf_blocks:
            x = block(x)
        x = self.lin_second(x)
        x = self.i_stft(x, spec, length)
        return x
