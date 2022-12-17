import torch

device = torch.device('cuda')
if not torch.cuda.is_available():
    device = torch.device('cpu')
    print('Not avaliable')


def to_cuda(data):
    if isinstance(data, (list, tuple)):
        return [to_cuda(x) for x in data]
    return data.to(device, non_blocking=True)


class CudaDataLoader:
    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for b in self.dl:
            yield to_cuda(b)

    def __len__(self):
        return len(self.dl)
