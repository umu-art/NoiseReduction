import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, size, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, size))  # [1, 1, size]
        self.beta = nn.Parameter(torch.Tensor(1, 1, size))  # [1, 1, size]
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x):
        """
        Args:
            x: [batch, T, size], bath is batch size, size is channel size, T is length
        Returns:
            gLN_y: [batch, T, size]
        """
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        var = torch.var(x, dim=(1, 2), unbiased=False, keepdim=True) + self.eps
        x = (x - mean) / torch.pow(var, 0.5)
        x = x * self.gamma + self.beta
        return x
