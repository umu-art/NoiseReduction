import librosa
import torch

from AudioMetods import read_audio
from model_BSS import Config
from model_BSS.ConformerBSS import ConformerBSS
from model_BSS.CudaDevice import to_cuda


class BSSModel:
    def __init__(self):
        self.model = ConformerBSS(Config.size, Config.conf_blocks_num, Config.conv_kernel_size, 1, Config.w_len,
                                  Config.w_len // 2)
        snap = torch.load('model_BSS/model_BSS.tar', map_location='cpu')
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
            audio = audio[:16_000 * 30]
            audio = to_cuda(audio)
            wave = self.model(audio[None].float())[0]
            return wave.cpu().numpy()
