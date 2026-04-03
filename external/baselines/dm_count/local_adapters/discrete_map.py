"""Discrete map helpers for DM-Count local dataset adaptation."""

from __future__ import annotations

import numpy as np
import torch


def _normalize_points(points) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        pts = points.detach().cpu().numpy()
    else:
        pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def generate_downsampled_discrete_map(points, image_shape, downsample_ratio: int = 8) -> torch.Tensor:
    pts = _normalize_points(points)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    ratio = int(downsample_ratio)

    if ratio <= 0:
        raise ValueError("downsample_ratio must be positive")
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_shape must be positive")
    if image_h % ratio != 0 or image_w % ratio != 0:
        raise ValueError("image_shape must be divisible by downsample_ratio for DM-Count discrete maps")

    down_h = image_h // ratio
    down_w = image_w // ratio
    discrete = torch.zeros((1, down_h, down_w), dtype=torch.float32)

    if pts.shape[0] == 0:
        return discrete

    max_x = np.nextafter(np.float32(image_w), np.float32(-np.inf))
    max_y = np.nextafter(np.float32(image_h), np.float32(-np.inf))
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, max_x)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, max_y)

    x_index = np.floor(pts[:, 0] / ratio).astype(np.int64)
    y_index = np.floor(pts[:, 1] / ratio).astype(np.int64)
    x_index = np.clip(x_index, 0, down_w - 1)
    y_index = np.clip(y_index, 0, down_h - 1)

    for x_coord, y_coord in zip(x_index, y_index):
        discrete[0, y_coord, x_coord] += 1.0

    return discrete
