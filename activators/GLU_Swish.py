import torch


class GLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor, dim: int):
        a, b = torch.chunk(x, 2, dim)
        return a * torch.sigmoid(b)


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.tensor):
        return x * torch.sigmoid(x)


if __name__ == '__main__':
    x = torch.rand(16, 512, 301)

    my_swish = Swish()
    torch_swish = torch.nn.SiLU()
    assert torch.allclose(my_swish(x), torch_swish(x))

    my_GLU = GLU()
    torch_GLU = torch.nn.GLU()
    torch_GLU.dim = 1
    assert torch.allclose(my_GLU(x, 1), torch_GLU(x))
