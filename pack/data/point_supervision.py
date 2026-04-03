"""Point-supervision helpers shared by counting datasets."""
from __future__ import annotations

import numpy as np
import torch

from .density_map import generate_attention_mask, generate_density_map


def normalize_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2)


def clip_points_to_image(points, image_shape) -> np.ndarray:
    pts = normalize_points(points)
    if pts.size == 0:
        return pts

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    max_x = np.nextafter(np.float32(w), np.float32(-np.inf))
    max_y = np.nextafter(np.float32(h), np.float32(-np.inf))
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, max_x)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, max_y)
    return pts


def apply_transform_with_points(image, points, transform):
    pts = clip_points_to_image(points, image.shape[:2])

    if transform is None:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image_tensor, pts

    aug = transform(
        image=image,
        keypoints=[tuple(map(float, point)) for point in pts.tolist()],
    )
    aug_image = aug["image"]
    aug_points = clip_points_to_image(aug.get("keypoints", []), aug_image.shape[-2:] if isinstance(aug_image, torch.Tensor) else aug_image.shape[:2])

    if not isinstance(aug_image, torch.Tensor):
        aug_image = torch.from_numpy(np.asarray(aug_image)).permute(2, 0, 1).float() / 255.0

    return aug_image, aug_points


def build_point_supervision(points, image_shape, sigma: float, attention_radius: int):
    pts = normalize_points(points)
    h, w = int(image_shape[0]), int(image_shape[1])

    density = torch.from_numpy(generate_density_map(pts, (h, w), sigma=sigma)).unsqueeze(0)
    att_mask = torch.from_numpy(generate_attention_mask(pts, (h, w), radius=attention_radius)).unsqueeze(0)
    points_tensor = torch.from_numpy(pts.copy())
    count_tensor = torch.tensor(float(len(pts)), dtype=torch.float32)

    return density, att_mask, points_tensor, count_tensor
