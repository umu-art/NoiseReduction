from random import uniform

from torch.utils.data import Dataset

from AudioMetods import calc_coefficient
from Config import clean_pattern, noise_pattern, part_duration
from datasets.AudioDataset import AudioDataset


class MixDataset(Dataset):
    def __init__(self, snr_range: tuple, steps_per_epoch, duration: int = part_duration,
                 clean_pattern_: str = clean_pattern, noise_pattern_: str = noise_pattern):
        super().__init__()
        print('Init clean_dataset')
        self.clean_dataset = AudioDataset(clean_pattern_, duration, steps_per_epoch, 'clean')
        print('Init noise_dataset')
        self.noise_dataset = AudioDataset(noise_pattern_, duration, steps_per_epoch, 'noise')
        self.snr_range = snr_range
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, item):
        clean = self.clean_dataset.get()
        noise = self.noise_dataset.get()
        snr = uniform(self.snr_range[0], self.snr_range[1])
        mix = clean + calc_coefficient(clean, noise, snr) * noise
        return mix, clean
