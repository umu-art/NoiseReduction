import librosa
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

from Config import part_frames


def read_audio_len(path: str):
    """
        Ф-я возвращает длину аудио
        Вход:
            path (str) - путь к файлу
        Выход:
            **output** - длина файла
    """
    return sf.SoundFile(path).frames


def read_audio(path: str):
    """
        Ф-я читает аудио
        Вход:
            path (str) - путь к файлу
        Выход:
            **output** - (само аудио, частота, продолжительность)
    """
    audio, samplerate = sf.read(path)
    duration = len(audio) / samplerate
    return audio, samplerate, duration


def read_part_audio(path: str, start: int, frames: int):
    """
        Ф-я читает отрывок аудио
        Вход:
            path (str) - путь к файлу
            start (int) - начало
            frames (int) - кол-во кадров
        Выход:
            **output** - (само аудио, частота, продолжительность)
        """
    return sf.read(path, start=start, frames=frames)[0]


def calc_energy(x: np.ndarray) -> float:
    """
        Ф-я считает энергию аудио в Дб
        Вход:
            x (tensor) - аудио
        Выход:
            **output** (float) - энергия в Дб
    """
    if x.ndim == 1:
        x = x[None]
    return np.log10(np.sum(x ** 2, axis=1) / x.shape[1]) * 10


def calc_snr(clean, noise):
    """
        Ф-я считает SNR (в Дб)
        Вход:
            clean (tensor) - чистая запись
            noise (tensor) - шум
        Выход:
            **output** (float) - SNR
    """
    return calc_energy(clean) - calc_energy(noise)


def calc_coefficient(clean: np.ndarray, noise: np.ndarray, snr: float) -> float:
    """
        Ф-я считает коэффициент, на который нужно умножить шум,
        чтоб полученная запись имела заданный SNR

        Вход:
            clean (tensor) - чистая запись
            noise (tensor) - шум
            snr (float) - нужный SNR (в Дб)
        Выход:
            **output** (float) - энергия в Дб
    """
    f = (calc_energy(clean) - calc_energy(noise) - snr) / 20
    return 10 ** f


def save_audio(file, x):
    """
        Ф-я сохраняет аудио
        Вход:
            file (str) - куда сохранить
            x (tensor) - что нужно сохранить
    """
    sf.write(file, x, 16_000)


def ashow(x, sr=part_frames):
    """
        Ф-я превращает аудио в спектограмму и выводит её.
        Вход:
            x (tensor) - аудио
            sr (int) - частота дискретизации

    """
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(x)), ref=np.max), sr=sr, x_axis='s', y_axis='hz')
    plt.grid()
    plt.show()
