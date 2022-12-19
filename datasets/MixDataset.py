from random import uniform

from torch.utils.data import Dataset

from AudioMetods import calc_coefficient
from Config import clean_pattern, noise_pattern, part_frames, prefix_root
from datasets.AudioDataset import AudioDataset


class MixDataset(Dataset):
    def __init__(self, snr_range: tuple, steps_per_epoch, frames: int = part_frames,
                 clean_pattern_: str = clean_pattern, noise_pattern_: str = noise_pattern):
        super().__init__()
        addition_clean = clean_pattern_[len(prefix_root):len(prefix_root) + 4].lower()
        addition_noise = noise_pattern_[len(prefix_root):len(prefix_root) + 4].lower()
        print('Init clean_dataset')
        self.clean_dataset = AudioDataset(clean_pattern_, frames, steps_per_epoch, 'clean_' + addition_clean)
        print('Init noise_dataset')
        self.noise_dataset = AudioDataset(noise_pattern_, frames, steps_per_epoch, 'noise_' + addition_noise)
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
