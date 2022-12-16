from torch import nn

from layers.IStftLayer import IStftLayer
from layers.StftLayer import StftLayer


class NetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = StftLayer()
        self.lstm = nn.LSTM(input_size=513, hidden_size=513, batch_first=True)
        self.lin = nn.Linear(513, 513)
        self.istft = IStftLayer()

    def forward(self, mixture):
        spec, mag = self.stft(mixture)
        r = self.lstm(mag)[0]
        mask = self.lin(r)
        wave = self.istft(mask, spec)
        return wave
