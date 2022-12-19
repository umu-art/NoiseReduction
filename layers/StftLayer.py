import torch


class StftLayer(torch.nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: str, return_complex: bool = True):
        super().__init__()
        """
        arguments:
        n_fft(int) Length of the FFT used
        hop_length(int) the distance between neighboring sliding window frames
        win_length(int) the size of window frame and STFT filter
        window(str) the optional window function
        return_complex(bool) whether to return a complex tensor, or a real tensor with an extra last dimension for the real and imaginary components.
        """
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
        """
        arguments:
        mix(torch.tensor) the input tensor
        returns: A tensor containing the STFT result
        """
        spec = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, return_complex=self.return_complex)
        mag = spec.abs()
        mag = mag.permute(0, 2, 1).to(torch.float32)
        return spec, mag
