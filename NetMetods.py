import math
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from AudioMetods import ashow, calc_c, read_audio, save_audio
from Config import SR
from CudaDevice import to_cuda


def train(model, optimizer, loss_fn, data_loader, epochs):
    model.train()
    print('Training...')
    for _ in tqdm(range(epochs)):
        for mixture, clean in data_loader:
            wave = model(mixture)
            loss = loss_fn(wave, clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())


def test(model, dataset):
    f = dataset.clean_speech_data_paths[randint(0, len(dataset.clean_speech_data_paths) - 1)]
    audio = read_audio(f)[0]
    # ashow(audio)

    f = dataset.noise_paths[randint(0, len(dataset.noise_paths) - 1)]
    noise = read_audio(f)[0]
    while len(noise) < len(audio):
        noise = np.concatenate((noise, noise))
    noise = noise[:len(audio)]

    mixture = audio + calc_c(audio, noise, 2) * noise
    save_audio('mix.wav', mixture)


    mixture = torch.from_numpy(mixture)
    add = math.ceil(mixture.shape[0] / SR) * SR - mixture.shape[0]
    tenz_add = torch.zeros([add])
    mixture = torch.cat([mixture, tenz_add])
    x_len = mixture.shape[0]
    mixture = mixture.reshape([x_len // SR, SR])
    mixture = to_cuda(mixture)

    model.test()
    wave = model(mixture)
    wave = wave.reshape([x_len])
    save_audio('wave.wav', wave.cpu().detach().numpy())
