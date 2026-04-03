"""Evaluation metrics aligned to manuscript."""
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cal_mae(pred_count, gt_count):
    pred = _to_numpy(pred_count)
    gt = _to_numpy(gt_count)
    return float(np.mean(np.abs(pred - gt)))


def cal_mse(pred_count, gt_count):
    pred = _to_numpy(pred_count)
    gt = _to_numpy(gt_count)
    return float(np.mean((pred - gt) ** 2))


def cal_mape(pred_count, gt_count):
    pred = _to_numpy(pred_count)
    gt = _to_numpy(gt_count)
    valid = np.abs(gt) > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.abs((pred[valid] - gt[valid]) / gt[valid])) * 100.0)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
