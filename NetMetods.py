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


def train_epoch(model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler, loss_fn: nn.Module, data_loader,
                point: int, gl_point: int, clip_val: int):
    model.train()
    train_snr = 0
    train_inp_snr = 0
    train_snr_i = 0
    train_loss = 0
    for mixture, clean in tqdm(data_loader):
        wave = model(mixture)
        loss = loss_fn(wave, clean)
        loss.backward()
        Logger.write_grad_norm(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)), point)
        optimizer.step()

        Logger.write_lr(scheduler.get_last_lr(), point)
        scheduler.step()
        optimizer.zero_grad()
        cur_snr = calc_snr(clean.detach().cpu().numpy(),
                           wave.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        inp_snr = calc_snr(clean.detach().cpu().numpy(),
                           mixture.detach().cpu().numpy() - clean.detach().cpu().numpy()).mean()
        train_snr += cur_snr
        train_inp_snr += inp_snr
        train_loss += loss.item()
        train_snr_i += (cur_snr - inp_snr)

        Logger.write_point('train', point, cur_snr, inp_snr, loss.item())
        point += 1
    n = len(data_loader)
    train_snr /= n
    train_loss /= n
    train_inp_snr /= n
    train_snr_i /= n

    Logger.write_epoch_point('train_epoch', gl_point, train_snr, train_inp_snr, train_snr_i, train_loss)
    return {"train_loss": train_loss,
            "train_snr": train_snr,
            "train_inp_snr": train_inp_snr,
            "train_snr_i": train_snr_i
            }


@torch.no_grad()
def val_epoch(model, data_loader, loss_fn, point: int, gl_point: int):
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

        Logger.write_point('eval', point, cur_snr, inp_snr, loss.item())
        point += 1
    n = len(data_loader)
    val_snr /= n
    val_loss /= n
    val_inp_snr /= n
    val_snr_i /= n
    Logger.write_epoch_point('eval_epoch', gl_point, val_snr, val_inp_snr, val_snr_i, val_loss)
    return {"val_loss": val_loss,
            "val_snr": val_snr,
            "val_inp_snr": val_inp_snr,
            "val_snr_i": val_snr_i}


def train(model, optimizer, scheduler, loss_fn, data_loader_train, data_loader_val, epochs, save_path, clip_val: int):
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
    for epoch in tqdm(range(epochs)):
        cur_train = train_epoch(model, optimizer, scheduler, loss_fn, data_loader_train, epoch * Config.iters_per_epoch, epoch, clip_val)
        cur_val = val_epoch(model, data_loader_val, loss_fn, epoch * Config.iters_per_epoch, epoch)

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
    f = dataset.clean_dataset.audio_paths[randint(0, len(dataset.clean_dataset.audio_paths) - 1)]
    audio = read_audio(f)[0]
    # ashow(audio)

    f = dataset.noise_dataset.audio_paths[randint(0, len(dataset.noise_dataset.audio_paths) - 1)]
    noise = read_audio(f)[0]
    while len(noise) < len(audio):
        noise = np.concatenate((noise, noise))
    noise = noise[:len(audio)]

    mixture = audio + calc_coefficient(audio, noise, 2) * noise
    Logger.save_audio(torch.from_numpy(mixture), 'mix_' + str(i))
    # ashow(mixture)

    mixture = torch.from_numpy(mixture)
    add = math.ceil(mixture.shape[0] / Config.part_frames) * Config.part_frames - mixture.shape[0]
    tenz_add = torch.zeros([add])
    mixture = torch.cat([mixture, tenz_add])
    mixture = to_cuda(mixture)

    model.eval()
    wave = model(mixture[None])[0]
    Logger.save_audio(wave.cpu().detach(), 'wave_' + str(i))
    # ashow(wave.cpu().detach().numpy())
