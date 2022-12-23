import torch
from torchmetrics.functional import scale_invariant_signal_noise_ratio as si_snr


def sea_snr(preds, targets):
    batch_size, _, t_size = preds.shape
    res_one = torch.mean(si_snr(preds, targets), dim=1)
    preds = preds.permute(1, 0, 2)
    a, b = preds[1], preds[0]
    preds = torch.cat((a, b)).reshape(2, batch_size, t_size).permute(1, 0, 2)
    res_two = torch.mean(si_snr(preds, targets), dim=1)
    res = -torch.maximum(res_one, res_two)
    return res
