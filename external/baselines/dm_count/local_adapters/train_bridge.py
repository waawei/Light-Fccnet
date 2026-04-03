"""Training bridge for running upstream DM-Count on local datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import build_dmcount_datasets
from .runner import dmcount_train_collate


def build_dmcount_dataloaders(config: dict, dataset_name: str, downsample_ratio: int = 8):
    datasets = build_dmcount_datasets(config, dataset_name=dataset_name, downsample_ratio=downsample_ratio)
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=dmcount_train_collate,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }


def run_train_batch(
    model,
    batch,
    optimizer,
    ot_loss,
    tv_loss,
    mae_loss,
    wot: float,
    wtv: float,
    device="cpu",
):
    images, points, gt_discrete = batch
    images = images.to(device)
    points = [item.to(device) for item in points]
    gt_discrete = gt_discrete.to(device)
    batch_size = images.size(0)
    gt_count = np.array([len(item) for item in points], dtype=np.float32)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs, outputs_normed = model(images)

    ot_term, _, _ = ot_loss(outputs_normed, outputs, points)
    ot_term = ot_term * float(wot)

    count_targets = torch.from_numpy(gt_count).float().to(device)
    count_term = mae_loss(outputs.sum(dim=(1, 2, 3)), count_targets)

    gt_count_tensor = count_targets.view(batch_size, 1, 1, 1)
    gt_discrete_normed = gt_discrete / (gt_count_tensor + 1e-6)
    tv_term = (
        tv_loss(outputs_normed, gt_discrete_normed).sum(dim=(1, 2, 3)) * count_targets
    ).mean() * float(wtv)

    total_loss = ot_term + count_term + tv_term
    total_loss.backward()
    optimizer.step()

    pred_count = outputs.view(batch_size, -1).sum(dim=1).detach().cpu().numpy()
    pred_err = pred_count - gt_count

    return {
        "loss": float(total_loss.item()),
        "ot_loss": float(ot_term.item()),
        "count_loss": float(count_term.item()),
        "tv_loss": float(tv_term.item()),
        "mae": float(np.mean(np.abs(pred_err))),
        "mse": float(np.mean(pred_err * pred_err)),
    }
