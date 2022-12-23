import math

import librosa
import torch

from AudioMetods import read_audio
from model_NR import Config
from model_NR.Conformer import Conformer
from model_NR.CudaDevice import to_cuda


class NRModel:
    def __init__(self):
        self.model = Conformer(Config.n_fft, Config.hop_length, Config.win_length, Config.window,
                               Config.size, Config.conf_blocks_num, Config.conv_kernel_size)

        snap = torch.load('model_NR/model_NR.tar', map_location='cpu')
        model_state_dict = snap['model']
        self.model.load_state_dict(model_state_dict)
        to_cuda(self.model)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            a, b, c = read_audio(args[0])
            if a.ndim > 1:
                a = a[:, 0]
            a = librosa.resample(a, orig_sr=b, target_sr=16_000)
            audio = torch.from_numpy(a)
            audio = audio[:16_000 * 30 * 10]
            audio_len = len(audio)
            part_num = math.ceil(len(audio) / 16000 / 30)
            addition = torch.zeros(part_num * 16_000 * 30 - len(audio))
            audio = torch.cat((audio, addition))
            audio = audio.reshape((part_num, 16_000 * 30))
            audio = to_cuda(audio)
            out = torch.tensor([])
            for u in audio:
                wave = self.model(u[None])[0]
                out = torch.cat((out, wave.cpu()))
            out = out[:audio_len]
            return out.numpy()
