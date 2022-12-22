import torch
from torch import nn


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor):
        """
        arguments:
        x(torch.tensor) input tensor
        returns: x ⊗ σ(x)
        """
        return x * torch.sigmoid(x)
