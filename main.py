import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Config import conf_blocks_num, n_fft, hop_length, win_length, window, size, conv_kernel_size
from CudaDevice import CudaDataLoader, to_cuda
from MixtureDataset import MixtureDataset
from NetMetods import train, test
from Conformer import Conformer

# wget https://www.openslr.org/resources/17/musan.tar.gz
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# tar -xf musan.tar.gz
# tar -xf train-clean-100.tar.gz

dataset = MixtureDataset(16000, (0, 10), 10)
dataset.clean_speech_data_paths = dataset.clean_speech_data_paths[:10]
dataset.noise_paths = dataset.noise_paths[:2]

data_loader_train = DataLoader(dataset, batch_size=10, shuffle=False)
data_loader_valid = DataLoader(dataset, batch_size=10, shuffle=False)

data_loader_train = CudaDataLoader(data_loader_train)
data_loader_valid = CudaDataLoader(data_loader_valid)

loss_fn = nn.L1Loss()

model = Conformer(n_fft, hop_length, win_length, window, size, conf_blocks_num, conv_kernel_size)
to_cuda(model)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)

train(model, optimizer, loss_fn, data_loader_train, data_loader_valid, 15*2, '/out.tar')

test(model, dataset)
