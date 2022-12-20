from random import uniform
import numpy as np

from torch.utils.data import Dataset

import Config
from AudioMetods import calc_coefficient
from Config import clean_pattern, noise_pattern, part_frames, prefix_root
from datasets.NoiseDataset import NoiseDataset
from datasets.CleanDataset import CleanDataset


class MixDataset(Dataset):
    def __init__(self, snr_range: tuple, steps_per_epoch, frames: int = part_frames,
                 clean_pattern_: str = clean_pattern, noise_pattern_: str = noise_pattern):
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

    def __len__(self):
        return self.steps_per_epoch

    def get(self, a, b):
        clean_one = a.get()
        clean_two = b.get()
        noise = self.noise.get()
        snr = uniform(self.snr_range[0], self.snr_range[1])
        mix_clean = clean_one + clean_two
        mix = mix_clean + calc_coefficient(mix_clean, noise, snr) * noise
        return mix, np.stack([clean_one, clean_two])

    def __getitem__(self, item):
        chance = uniform(0, 1)
        if chance <= Config.chance_same_gender:
            chance = uniform(0, 1)
            if chance <= 0.5:
                return self.get(self.clean_man, self.clean_man)
            return self.get(self.clean_woman, self.clean_woman)
        chance = uniform(0, 1)
        if chance <= 0.5:
            chance = uniform(0, 1)
            if chance <= 0.5:
                return self.get(self.clean_woman, self.clean_man)
            return self.get(self.clean_man, self.clean_woman)



