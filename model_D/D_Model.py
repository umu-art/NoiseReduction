import math

import librosa
import torch

from AudioMetods import read_audio
from model_D import Config
from model_D.ConformerDetector import ConformerDetector


class DModel:
    def __init__(self):
        self.model = ConformerDetector(Config.size, Config.conf_blocks_num, Config.conv_kernel_size, 1, Config.w_len,
                                       Config.w_len // 2)
        snap = torch.load('model_D/model.tar', map_location='cpu')
        model_state_dict = snap['model']
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            a, b, c = read_audio(args[0])
            if a.ndim > 1:
                a = a[:, 0]
            a = librosa.resample(a, orig_sr=b, target_sr=16_000)
            audio = torch.from_numpy(a)
            audio = audio[:16_000 * 15 * 10]
            part_num = math.ceil(len(audio) / 16000 / 15)
            addition = torch.zeros(part_num * 16_000 * 15 - len(audio))
            audio = torch.cat((audio, addition))
            audio = audio.reshape((part_num, 16_000 * 15))
            two_speakers = False
            for u in audio:
                logit = self.model(u[None].float())[0]
                prob = torch.sigmoid(logit).item()
                print(prob)
                two_speakers = two_speakers | (prob > 0.5)
                if two_speakers:
                    break

            return two_speakers
