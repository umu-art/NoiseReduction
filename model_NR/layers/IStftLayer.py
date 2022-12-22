import torch
from torch import nn


class IStftLayer(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: str, return_complex: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.return_complex = return_complex
        if window == 'hann_window':
            self.register_buffer('window', torch.hann_window(n_fft))
        elif window == 'hamming_window':
            self.register_buffer('window', torch.hamming_window(n_fft))
        else:
            print('wrong argument "window"')
            raise

    """
        Аргументы: 
            x (torch.tensor): входные данные
        Возвращает:
            out (torch.tensor): данный массив, прогнанный через все нужные слои
    """

    def forward(self, mask, spec, length: int):
        mask = mask.permute(0, 2, 1)
        spec_estimate = mask * spec
        wave_estimate = torch.istft(spec_estimate, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.win_length, window=self.window, length=length)
        return wave_estimate
