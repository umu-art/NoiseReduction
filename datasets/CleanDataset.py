import os
from pathlib import Path
from random import randint

from tqdm import tqdm

from AudioMetods import read_audio_len, read_part_audio
from datasets.CacheDatasetPR import exist_cache, read_cache, save_cache


class CleanDataset:
    def __init__(self, path_to_data: str, count_frames: int, gender: str, cache_name: str):
        if exist_cache(f'{cache_name}_{gender}_audio_paths.vc') and exist_cache(f'{cache_name}_{gender}_count_frames.vc'):
            print('Load from cache')
            self.audio_paths = read_cache(f'{cache_name}_{gender}_audio_paths.vc')
            self.count_frames = [int(u) for u in read_cache(f'{cache_name}_{gender}_count_frames.vc')]
        else:
            path_to_info = Path(path_to_data).parent.joinpath('SPEAKERS.TXT')
            info = path_to_info.open('r')
            info_data = [u.split()[0] for u in info.readlines() if not u.startswith(';') and f'| {gender} |' in u]
            available_by_gender = set(info_data)
            self.audio_paths = []
            self.count_frames = []
            for speaker in tqdm(os.listdir(path_to_data)):
                if speaker in available_by_gender:
                    for book in os.listdir(Path(path_to_data).joinpath(speaker).absolute()):
                        for part in os.listdir(Path(path_to_data).joinpath(speaker).joinpath(book).absolute()):
                            if part.endswith('wav') or part.endswith('flac'):
                                path = Path(path_to_data).joinpath(speaker).joinpath(book).joinpath(part).absolute()
                                audio_frames = read_audio_len(str(path))
                                if audio_frames >= count_frames:
                                    self.audio_paths.append(path)
                                    self.count_frames.append(audio_frames)
            assert len(self.audio_paths) > 0
            save_cache(self.audio_paths, f'{cache_name}_{gender}_audio_paths.vc')
            save_cache(self.count_frames, f'{cache_name}_{gender}_count_frames.vc')
        self.chunk_size = count_frames
        self.last = -1

    def get(self):
        while True:
            try:
                index = randint(0, len(self.audio_paths) - 1)
                while self.last == index:
                    index = randint(0, len(self.audio_paths) - 1)

                start = randint(0, self.count_frames[index] - self.chunk_size)
                audio = read_part_audio(self.audio_paths[index], start, self.chunk_size)
                return audio
            finally:
                pass
