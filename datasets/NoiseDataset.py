from glob import glob
from random import randint

from tqdm import tqdm

from AudioMetods import read_part_audio, read_audio_len
from datasets.CacheDatasetPR import exist_cache, read_cache, save_cache


class NoiseDataset:

    def __init__(self, path: str, count_frames: int, cache_name: str):
        all_audio_paths = glob(path, recursive=True)

        if exist_cache(cache_name + '_audio_paths.vc') and exist_cache(cache_name + '_count_frames.vc'):
            print('Load from cache')
            self.audio_paths = read_cache(cache_name + '_audio_paths.vc')
            self.count_frames = [int(u) for u in read_cache(cache_name + '_count_frames.vc')]
        else:
            self.audio_paths = []
            self.count_frames = []
            for audio_path in tqdm(all_audio_paths):
                audio_len = read_audio_len(audio_path)
                if audio_len >= count_frames:
                    self.audio_paths.append(audio_path)
                    self.count_frames.append(audio_len)
            save_cache(self.audio_paths, cache_name + '_audio_paths.vc')
            save_cache(self.count_frames, cache_name + '_count_frames.vc')

        assert len(self.audio_paths) > 1

        self.chunk_size = count_frames

    def get(self):
        while True:
            try:
                index = randint(0, len(self.audio_paths) - 1)
                start = randint(0, self.count_frames[index] - self.chunk_size)
                audio = read_part_audio(self.audio_paths[index], start, self.chunk_size)
                return audio
            except:
                pass
