from torch import nn


class ResCon(nn.Module):
    def __init__(self, a, b, module):
        super().__init__()
        self.a = a
        self.b = b
        self.module = module

    def forward(self, x):
        return self.a * x + self.b * self.module(x)
