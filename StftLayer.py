import torch


class StftLayer(torch.nn.Module):
    def __init__(self,n_fft:int, hop_length:int, win_length:int, window:str, return_complex:bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.return_complex = return_complex
        if window == 'hann_window':
            self.register_buffer('window', torch.hann_window(n_fft))
        elif window == 'hamming_window':
            self.register_buffer('window', torch.hamming_window(n_fft))
        else:
            print('wrong argument "window"')
            raise

    def forward(self, mix):
        spec = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=self.return_complex)
        mag = spec.abs()
        mag = mag.permute(0, 2, 1).to(torch.float32)
        return spec, mag

if __name__ == '__main__':
    x = torch.rand(16,16000)
    a = torch.stft(x,n_fft= 100,hop_length= 100,win_length= 100,window= torch.hann_window(100), return_complex= True)
    b = StftLayer(n_fft= 100,hop_length= 100,win_length= 100,window= 'hann_window', return_complex= True)
    assert a!=b(x)
