"""Training bridge for running upstream SASNet on local datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import build_sasnet_datasets
from .runner import sasnet_train_collate


def build_sasnet_dataloaders(config: dict, dataset_name: str):
    datasets = build_sasnet_datasets(config, dataset_name=dataset_name)
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
            collate_fn=sasnet_train_collate,
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
    criterion,
    device="cpu",
    log_para: float = 1000.0,
):
    images, density_maps = batch
    images = images.to(device)
    density_maps = density_maps.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)

    density_scale = float(log_para) if float(log_para) != 0.0 else 1.0
    scaled_targets = density_maps * density_scale
    density_loss = criterion(outputs, scaled_targets)
    density_loss.backward()
    optimizer.step()

    pred_count = outputs.view(outputs.size(0), -1).sum(dim=1).detach().cpu().numpy() / density_scale
    gt_count = density_maps.view(density_maps.size(0), -1).sum(dim=1).detach().cpu().numpy()
    pred_err = pred_count - gt_count

    return {
        "loss": float(density_loss.item()),
        "density_loss": float(density_loss.item()),
        "mae": float(np.mean(np.abs(pred_err))),
        "mse": float(np.mean(pred_err * pred_err)),
    }
