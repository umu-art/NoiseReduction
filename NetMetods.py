import math
from random import randint

import numpy as np
import torch
from tqdm.auto import tqdm

import Config
from AudioMetods import calc_coefficient, read_audio, calc_snr, save_audio
from CudaDevice import to_cuda


def train_epoch(model, optimizer, loss_fn, data_loader):
    model.train()
    train_snr = 0
    train_inp_snr = 0
    train_snr_i = 0
    train_loss = 0
    for mixture, clean in tqdm(data_loader):
        wave = model(mixture)
        loss = loss_fn(wave, clean)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_snr = calc_snr(clean.detach().cpu().numpy(),
                           wave.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        inp_snr = calc_snr(clean.detach().cpu().numpy(),
                           mixture.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        train_snr += cur_snr
        train_inp_snr += inp_snr
        train_loss += loss.item()
        train_snr_i += (cur_snr - inp_snr)
    n = len(data_loader)
    train_snr /= n
    train_loss /= n
    train_inp_snr /= n
    train_snr_i /= n
    return {"train_loss": train_loss,
            "train_snr": train_snr,
            "train_inp_snr": train_inp_snr,
            "train_snr_i": train_snr_i
            }


@torch.no_grad()
def val_epoch(model, data_loader, loss_fn):
    model.eval()
    val_snr = 0
    val_inp_snr = 0
    val_loss = 0
    val_snr_i = 0
    for mixture, clean in data_loader:
        wave = model(mixture)
        loss = loss_fn(wave, clean)
        cur_snr = calc_snr(clean.detach().cpu().numpy(),
                           wave.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        inp_snr = calc_snr(clean.detach().cpu().numpy(),
                           mixture.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        val_snr += cur_snr
        val_inp_snr += inp_snr
        val_loss += loss.item()
        val_snr_i += (cur_snr - inp_snr)
    n = len(data_loader)
    val_snr /= n
    val_loss /= n
    val_inp_snr /= n
    val_snr_i /= n
    return {"val_loss": val_loss,
            "val_snr": val_snr,
            "val_inp_snr": val_inp_snr,
            "val_snr_i": val_snr_i}


def train(model, optimizer, loss_fn, data_loader_train, data_loader_val, epochs, save_path):
    # save_path .tar
    logs = {
        "train_loss": [],
        "train_snr": [],
        "train_inp_snr": [],
        "train_snr_i": [],
        "val_loss": [],
        "val_snr": [],
        "val_inp_snr": [],
        "val_snr_i": [],
    }

    print('Training...')
    for _ in tqdm(range(epochs)):
        cur_train = train_epoch(model, optimizer, loss_fn, data_loader_train)
        cur_val = val_epoch(model, data_loader_val, loss_fn)

        logs["train_loss"].append(cur_train["train_loss"])
        logs["train_snr"].append(cur_train["train_snr"])
        logs["train_inp_snr"].append(cur_train["train_inp_snr"])
        logs["train_snr_i"].append(cur_train["train_snr_i"])

        logs["val_loss"].append(cur_val["val_loss"])
        logs["val_snr"].append(cur_val["val_snr"])
        logs["val_inp_snr"].append(cur_val["val_inp_snr"])
        logs["val_snr_i"].append(cur_val["val_snr_i"])
        snapshot = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "logs": logs,
        }
        print('Loss', cur_val["val_loss"])

        torch.save(snapshot, save_path)


def test(model, dataset, i):
    f = dataset.clean_speech_data_paths[randint(0, len(dataset.clean_speech_data_paths) - 1)]
    audio = read_audio(f)[0]
    # ashow(audio)

    f = dataset.noise_paths[randint(0, len(dataset.noise_paths) - 1)]
    noise = read_audio(f)[0]
    while len(noise) < len(audio):
        noise = np.concatenate((noise, noise))
    noise = noise[:len(audio)]

    mixture = audio + calc_coefficient(audio, noise, 2) * noise
    save_audio('/content/out/mix' + str(i) + '.wav', mixture)
    # ashow(mixture)

    mixture = torch.from_numpy(mixture)
    add = math.ceil(mixture.shape[0] / Config.part_frames) * Config.part_frames - mixture.shape[0]
    tenz_add = torch.zeros([add])
    mixture = torch.cat([mixture, tenz_add])
    x_len = mixture.shape[0]
    mixture = mixture.reshape([x_len // Config.part_frames, Config.part_frames])
    mixture = to_cuda(mixture)

    model.eval()
    wave = model(mixture)
    wave = wave.reshape([x_len])
    save_audio('/content/out/wave' + str(i) + '.wav', wave.cpu().detach().numpy())
    # ashow(wave.cpu().detach().numpy())
