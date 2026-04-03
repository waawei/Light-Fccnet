"""LDMS helpers for Light-FCCNet."""
from __future__ import annotations

import math

import torch


def _as_points_tensor(points) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        pts = points.detach().float()
    else:
        pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.numel() == 0:
        return pts.reshape(0, 2)
    return pts.reshape(-1, 2)


def compute_ldms_scales(
    points,
    image_shape,
    k: int = 3,
    factor: float = 1.0,
    max_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Compute LDMS local scales using K-nearest-neighbor distances.

    d_i = min(f * mean(KNN_i), max_ratio * min(H, W))
    """
    pts = _as_points_tensor(points)
    if pts.shape[0] == 0:
        return torch.zeros(0, dtype=torch.float32)

    h, w = int(image_shape[0]), int(image_shape[1])
    max_scale = float(min(h, w)) * float(max_ratio)
    if pts.shape[0] == 1:
        return torch.full((1,), max_scale, dtype=torch.float32)

    pairwise = torch.cdist(pts, pts, p=2)
    pairwise.fill_diagonal_(float("inf"))
    k = max(1, min(int(k), pts.shape[0] - 1))
    knn_dist, _ = torch.topk(pairwise, k=k, largest=False, dim=1)
    scales = float(factor) * knn_dist.mean(dim=1)
    return torch.clamp(scales, max=max_scale)


def compute_match_thresholds(box_width: float = 4.0, box_height: float = 8.0) -> tuple[float, float]:
    """
    Paper text gives fixed annotated box width/height w=4, h=8.
    Use:
      delta_x = min(w, h) / 2
      delta_y = sqrt(w^2 + h^2) / 2
    """
    w = float(box_width)
    h = float(box_height)
    delta_x = min(w, h) / 2.0
    delta_y = math.sqrt(w * w + h * h) / 2.0
    return delta_x, delta_y
