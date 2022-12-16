import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CudaDevice import CudaDataLoader, to_cuda
from MixtureDataset import MixtureDataset
from NetMetods import train, test
from NetModel import NetModel

SR = 16000

clean_speech_data_root = '/content/LibriSpeech/train-clean-100/'
noise_root = 'E:/musan/noise'

dataset = MixtureDataset(16000, (0, 10), 10000)
dataset.clean_speech_data_paths = dataset.clean_speech_data_paths[:10000]
dataset.noise_paths = dataset.noise_paths[:10]

data_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
data_loader = CudaDataLoader(data_loader)

loss_fn = nn.L1Loss()

model = NetModel()
to_cuda(model)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3)

train(model, optimizer, loss_fn, data_loader, epchs=30)

test(model, dataset)
