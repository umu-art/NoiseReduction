from torch import nn

from activators.Swish import Swish


class FeedForwardModule(nn.Module):
    def __init__(self, size: int, dropout_p: float, expansion_factor: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.lin_first = nn.Linear(size, size * expansion_factor)
        self.swish = Swish()
        self.dropout_first = nn.Dropout(p=dropout_p)
        self.lin_second = nn.Linear(size * expansion_factor, size)
        self.dropout_second = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.lin_first(x)
        x = self.swish(x)
        x = self.dropout_first(x)
        x = self.lin_second(x)
        x = self.dropout_second(x)
        return x
