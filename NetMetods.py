import math
from random import randint

import numpy as np
import torch
from torch import optim, nn
from tqdm.auto import tqdm

import Config
import Logger
from AudioMetods import calc_coefficient, read_audio, calc_snr
from CudaDevice import to_cuda


def test(model, dataset, i: str):
    f = dataset.clean_dataset.audio_paths[randint(0, len(dataset.clean_dataset.audio_paths) - 1)]
    audio = read_audio(f)[0]
    # ashow(audio)

    f = dataset.noise_dataset.audio_paths[randint(0, len(dataset.noise_dataset.audio_paths) - 1)]
    noise = read_audio(f)[0]
    while len(noise) < len(audio):
        noise = np.concatenate((noise, noise))
    noise = noise[:len(audio)]

    mixture = audio + calc_coefficient(audio, noise, 2) * noise
    Logger.save_audio(torch.from_numpy(mixture), str(i) + '_mix')
    # ashow(mixture)

    mixture = torch.from_numpy(mixture)
    add = math.ceil(mixture.shape[0] / Config.part_frames) * Config.part_frames - mixture.shape[0]
    tenz_add = torch.zeros([add])
    mixture = torch.cat([mixture, tenz_add])
    mixture = to_cuda(mixture)

    model.eval()
    wave = model(mixture[None])[0]
    Logger.save_audio(wave.cpu().detach(), str(i) + '_wave')
    # ashow(wave.cpu().detach().numpy())
