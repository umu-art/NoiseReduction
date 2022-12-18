from glob import glob
from random import randint

from tqdm import tqdm

from AudioMetods import read_audio, read_part_audio, read_audio_len
from datasets.CacheDatasetPR import exist_cache, read_cache, save_cache


class AudioDataset:

    def __init__(self, path: str, duration: int, steps_per_epoch: int, cache_name: str):
        super().__init__()
        all_audio_paths = glob(path, recursive=True)

        if exist_cache(cache_name + '_audio_paths.vc') and exist_cache(cache_name + '_audio_lens.vc'):
            print('Load from cache')
            self.audio_paths = read_cache(cache_name + '_audio_paths.vc')
            self.audio_lens = [int(u) for u in read_cache(cache_name + '_audio_lens.vc')]
        else:
            self.audio_paths = []
            self.audio_lens = []
            for audio_path in tqdm(all_audio_paths):
                audio_len = read_audio_len(audio_path)
                if audio_len >= duration:
                    self.audio_paths.append(audio_path)
                    self.audio_lens.append(audio_len)
            save_cache(self.audio_paths, cache_name + '_audio_paths.vc')
            save_cache(self.audio_lens, cache_name + '_audio_lens.vc')

        assert len(self.audio_paths) > 0

        self.chunk_size = duration
        self.steps_per_epoch = steps_per_epoch

    def get(self):
        index = randint(0, len(self.audio_paths) - 1)
        start = randint(0, self.audio_lens[index] - self.chunk_size)
        audio = read_part_audio(self.audio_paths[index], start, self.chunk_size)
        return audio





