import librosa
import torch

from AudioMetods import read_audio
from model import Config
from model.Conformer import Conformer


class NRModel:
    def __init__(self):
        self.model = Conformer(Config.n_fft, Config.hop_length, Config.win_length, Config.window,
                          Config.size, Config.conf_blocks_num, Config.conv_kernel_size)

        snap = torch.load('model/model.tar', map_location='cpu')
        model_state_dict = snap['model']
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            a, b, c = read_audio(args[0])
            a = a[:, 0]
            a = librosa.resample(a, orig_sr=b, target_sr=16_000)
            audio = torch.from_numpy(a)[None]
            wave = self.model(audio)[0]
            return wave.numpy()
