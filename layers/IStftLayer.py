import torch
from torch import nn


class IStftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(1024))

    def forward(self, mask, spec):
        mask = mask.permute(0, 2, 1)
        spec_estimate = mask * spec
        wave_estimate = torch.istft(spec_estimate, 1024, hop_length=512, win_length=1024, window=self.window, length=SR)
        return wave_estimate
