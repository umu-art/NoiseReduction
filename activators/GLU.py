import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self, dim):
        """
        arguments:
        dim(int) the dimension on which to split the input
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor):
        """
        arguments:
        x(torch.tensor) input tensor
        returns: a ⊗ σ(b) where a is the first half of the input matrices and b is the second half
        """
        a, b = torch.chunk(x, 2, self.dim)
        return a * torch.sigmoid(b)
