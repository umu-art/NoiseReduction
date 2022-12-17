import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CudaDevice import CudaDataLoader, to_cuda
from MixtureDataset import MixtureDataset
from NetMetods import train, test
from NetModel import NetModel

# wget https://www.openslr.org/resources/17/musan.tar.gz
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# tar -xf musan.tar.gz
# tar -xf train-clean-100.tar.gz

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
