import torch

import Config
from AudioMetods import read_audio, save_audio
from model.Conformer import Conformer
import librosa

# wget https://www.openslr.org/resources/17/musan.tar.gz
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# tar -xf musan.tar.gz
# tar -xf train-clean-100.tar.gz

if __name__ == '__main__':
    print(torch.__version__)

    model = Conformer(Config.n_fft, Config.hop_length, Config.win_length, Config.window,
                      Config.size, Config.conf_blocks_num, Config.conv_kernel_size)
    snap = torch.load('model.tar', map_location='cpu')
    model_state_dict = snap['model']
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        a, b, c = read_audio('music.mp3')
        a = a[:,0]
        a = librosa.resample(a, orig_sr=44100, target_sr=16_000)
        a = a[:16000 * 60]
        audio = torch.from_numpy(a)[None]

        wave = model(audio)[0]
        save_audio('./out.wav', wave.numpy())
