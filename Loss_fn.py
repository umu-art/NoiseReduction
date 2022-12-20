import torch
from Config import snr


def si_snr(preds, targets):
    ans = []
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        res_one = snr(pred, target)
        pred[0], pred[1] = pred[1], pred[0]
        res_two = snr(pred, target)
        ans.append(-max(res_one, res_two))
    ans = torch.tensor(ans)
    return ans