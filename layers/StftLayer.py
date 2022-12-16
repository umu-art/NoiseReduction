import torch
from torch import nn


class StftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(1024))

    def forward(self, mix):
        spec = torch.stft(mix, 1024, hop_length=512, win_length=1024, window=self.window, return_complex=True)
        mag = spec.abs()
        mag = mag.permute(0, 2, 1).to(torch.float32)
        return spec, mag
