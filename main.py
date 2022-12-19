import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import Config
from CudaDevice import CudaDataLoader, to_cuda
from NetMetods import train, test
from Conformer import Conformer
from datasets.MixDataset import MixDataset

# wget https://www.openslr.org/resources/17/musan.tar.gz
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# tar -xf musan.tar.gz
# tar -xf train-clean-100.tar.gz

if __name__ == 'main':
    print(torch.__version__)

    dataset = MixDataset(Config.snr_range, Config.iters_per_epoch * Config.batch_size)

    data_loader_train = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)
    data_loader_valid = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    data_loader_train = CudaDataLoader(data_loader_train)
    data_loader_valid = CudaDataLoader(data_loader_valid)

    loss_fn = nn.L1Loss()

    model = Conformer(Config.n_fft, Config.hop_length, Config.win_length, Config.window,
                      Config.size, Config.conf_blocks_num, Config.conv_kernel_size, Config.length)

    to_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), betas=Config.betas, lr=Config.lr)

    train(model, optimizer, loss_fn, data_loader_train, data_loader_valid, Config.epochs, Config.save_path)

    for i in range(10):
        test(model, dataset, i)

    while True:
        if input() == 'finish':
            break
