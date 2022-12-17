import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor):
        a, b = torch.chunk(x, 2, self.dim)
        return a * torch.sigmoid(b)
