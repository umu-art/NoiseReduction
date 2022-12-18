import librosa
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

from Config import part_duration


def read_audio_len(path: str):
    return sf.SoundFile(path).frames


def read_audio(path: str):
    audio, samplerate = sf.read(path)
    duration = len(audio) / samplerate
    return audio, samplerate, duration


def read_part_audio(path: str, start: int, duration: int):
    return sf.read(path, start=start, frames=duration)[0]


def calc_energy(x: np.ndarray) -> float:
    if x.ndim == 1:
        x = x[None]
    return np.log10(np.sum(x ** 2, axis=1) / x.shape[1]) * 10


def calc_snr(clean, noise):
    return calc_energy(clean) - calc_energy(noise)


def calc_coefficient(clean: np.ndarray, noise: np.ndarray, snr: float) -> float:
    f = (calc_energy(clean) - calc_energy(noise) - snr) / 20
    return 10 ** f


def save_audio(file, x):
    sf.write(file, x, part_duration)


def ashow(x, sr=part_duration):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(x)), ref=np.max), sr=sr, x_axis='s', y_axis='hz')
    plt.grid()
    plt.show()
