from random import uniform
import numpy as np

from torch.utils.data import Dataset

import Config
from AudioMetods import calc_coefficient
from Config import clean_pattern, noise_pattern, part_frames, prefix_root
from datasets.NoiseDataset import NoiseDataset
from datasets.CleanDataset import CleanDataset


def calc_energy(x: np.ndarray, eps: float = 1e-8) -> float:
    if x.ndim == 1:
        x = x[None]
    return np.log10(np.sum(x ** 2, axis=1) / x.shape[1] + eps) * 10


def normalize(x: np.ndarray, level: float, eps: float = 1e-8) -> np.ndarray:
    if x.ndim == 1:
        x = x[None]
    summ = 10 ** (level / 10) * x.shape[1]
    coef = summ / (np.sum(x ** 2, axis=1)[0] + eps)
    return x * np.sqrt(coef)


class Normalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray):
        # размерность только одна (по договоренности)
        level = np.random.normal(loc=self.mean, scale=self.std, size=1)
        return normalize(x, level)


class MixDataset(Dataset):
    def __init__(self, snr_range: tuple, steps_per_epoch, frames: int = part_frames,
                 clean_pattern_: str = clean_pattern, noise_pattern_: str = noise_pattern,
                 mean: float = -23, level: float = 3):
        super().__init__()
        addition_clean = clean_pattern_[len(prefix_root):len(prefix_root) + 4].lower()
        addition_noise = noise_pattern_[len(prefix_root):len(prefix_root) + 4].lower()
        print('Init clean_dataset')
        self.clean_man = CleanDataset(Config.clean_speech_data_root, frames, 'M', 'clean_' + addition_clean)
        self.clean_woman = CleanDataset(Config.clean_speech_data_root, frames, 'F', 'clean_' + addition_clean)
        print('Init noise_dataset')
        self.noise = NoiseDataset(noise_pattern_, frames, 'noise_' + addition_noise)
        self.snr_range = snr_range
        self.steps_per_epoch = steps_per_epoch
        self.nm = Normalizer(mean, level)

    def __len__(self):
        return self.steps_per_epoch

    def get_two_people(self, a, b):
        clean_one = a.get()
        clean_two = b.get()

        clean_one = self.nm(clean_one)
        clean_two = self.nm(clean_two)
        mix_clean = clean_one + clean_two
        chance = uniform(0, 1)
        if chance <= Config.chance_noise:
            noise = self.noise.get()
            snr = uniform(self.snr_range[0], self.snr_range[1])
            mix = mix_clean + calc_coefficient(mix_clean, noise, snr) * noise
        else:
            mix = mix_clean
        return mix[0].astype('float32')

    def get_one_people(self, a):
        clean_one = a.get()
        clean_one = self.nm(clean_one)
        chance = uniform(0, 1)
        if chance <= Config.chance_noise:
            noise = self.noise.get()
            snr = uniform(self.snr_range[0], self.snr_range[1])
            mix = clean_one + calc_coefficient(clean_one, noise, snr) * noise
        else:
            mix = clean_one
        return mix[0].astype('float32')

    def __getitem__(self, item):
        chance = uniform(0, 1)
        if chance <= Config.chance_one_speaker:
            chance = uniform(0, 1)
            if chance <= 0.5:
                return self.get_one_people(self.clean_man), 0
            else:
                return self.get_one_people(self.clean_woman), 0
        else:
            chance = uniform(0, 1)
            if chance <= Config.chance_different_gender:
                return self.get_two_people(self.clean_man, self.clean_woman), 1
            else:
                chance = uniform(0, 1)
                if chance <= 0.5:
                    return self.get_two_people(self.clean_man, self.clean_man), 1
                else:
                    return self.get_two_people(self.clean_woman, self.clean_woman), 1

