from glob import glob
from random import randint, uniform
from torch.utils.data import Dataset
from tqdm import tqdm

from AudioMetods import read_audio, calc_c
from Config import clean_speech_data_root, noise_root


class MixtureDataset(Dataset):
    def get_random_clean(self):
        f = self.clean_speech_data_paths[randint(0, len(self.clean_speech_data_paths) - 1)]
        audio = read_audio(f)[0]
        st = randint(0, len(audio) - self.chunk_size)
        return audio[st:st + self.chunk_size]

    def get_random_noise(self):
        f = self.noise_paths[randint(0, len(self.noise_paths) - 1)]
        audio = read_audio(f)[0]
        st = randint(0, len(audio) - self.chunk_size)
        return audio[st:st + self.chunk_size]

    def __init__(self, chunk_size, snr_range, n_steps_per_epoch):
        all_clean_speech_data_paths = glob(f'{clean_speech_data_root}/**/*.flac', recursive=True)[:5000] # TODO: error here
        all_noise_paths = glob(f'{noise_root}/**/*.wav', recursive=True)

        print('Init clean_speech_data')
        self.clean_speech_data_paths = [u for u in tqdm(all_clean_speech_data_paths) if
                                        len(read_audio(u)[0]) >= chunk_size]
        print('Init noise_data')
        self.noise_paths = [u for u in tqdm(all_noise_paths) if len(read_audio(u)[0]) >= chunk_size]

        assert len(self.clean_speech_data_paths) > 0
        assert len(self.noise_paths) > 0

        self.chunk_size = chunk_size
        self.snr_range = snr_range
        self.n_steps_per_epoch = n_steps_per_epoch

    def __len__(self) -> int:
        return self.n_steps_per_epoch

    def __getitem__(self, idx: int):
        clean = self.get_random_clean()
        noise = self.get_random_noise()
        snr = uniform(self.snr_range[0], self.snr_range[1])
        mixture = clean + calc_c(clean, noise, snr) * noise
        return mixture, clean
