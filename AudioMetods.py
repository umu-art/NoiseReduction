import librosa
import numpy as np
from IPython.core.display_functions import display
from IPython.lib.display import Audio
import soundfile as sf
from matplotlib import pyplot as plt

from Config import SR


def read_audio(path: str):
    audio, samplerate = sf.read(path)
    duration = len(audio) / samplerate
    return audio, samplerate, duration


def calc_energy(x: np.ndarray) -> float:
    if x.ndim == 1:
        x = x[None]
    return np.log10(np.sum(x ** 2, axis=1) / x.shape[1]) * 10


def calc_c(clean: np.ndarray, noise: np.ndarray, snr: float) -> float:
    f = (calc_energy(clean) - calc_energy(noise) - snr) / 20
    return 10 ** f


def adisplay(x, sr=SR):
    display(Audio(x, rate=sr))


def save_audio(file, x):
    sf.write(file, x, SR)


def ashow(x, sr=SR):
    adisplay(x, sr)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(x)), ref=np.max), sr=sr, x_axis='s', y_axis='hz')
    plt.grid()
    plt.show()
