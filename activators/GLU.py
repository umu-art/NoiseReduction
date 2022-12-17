import torch


class GLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor, dim: int):
        a, b = torch.chunk(x, 2, dim)
        return a * torch.sigmoid(b)
