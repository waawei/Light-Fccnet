"""Density target helpers for local SASNet adaptation."""

from __future__ import annotations

import numpy as np
import torch

from pack.data import generate_density_map


def build_sasnet_density_target(
    density: torch.Tensor | np.ndarray | None,
    points: torch.Tensor | np.ndarray | None,
    image_shape: tuple[int, int],
    sigma: int = 4,
) -> torch.Tensor:
    if density is not None:
        if isinstance(density, torch.Tensor):
            tensor = density.detach().clone().float()
        else:
            tensor = torch.as_tensor(density, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError("density target must have shape [H, W] or [1, H, W]")
        return tensor

    if points is None:
        point_array = np.zeros((0, 2), dtype=np.float32)
    elif isinstance(points, torch.Tensor):
        point_array = points.detach().cpu().numpy().reshape(-1, 2).astype(np.float32)
    else:
        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    density_np = generate_density_map(point_array, image_shape=image_shape, sigma=sigma)
    return torch.from_numpy(density_np).unsqueeze(0).float()

