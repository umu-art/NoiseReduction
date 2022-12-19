import numpy as np
import torch


class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=3, warmup_lr_init=1e-5,
                 min_lr=1e-5,
                 last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_epochs:
            return list(self.last_epoch / self.warmup_epochs * np.array(self.base_lrs) + self.warmup_lr_init)
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - self.warmup_epochs) % self.step_size:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.warmup_epochs) % self.step_size == 0:
            r = [0] * len([group['lr'] for group in self.optimizer.param_groups])
            for i in range(len(r)):
                r[i] = [group['lr'] for group in self.optimizer.param_groups][i] * self.gamma
                if r[i] < self.min_lr:
                    r[i] = [group['lr'] for group in self.optimizer.param_groups][i]
            return r

        raise NotImplementedError
