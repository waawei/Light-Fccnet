"""Density map generation utilities (paper-aligned)."""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def generate_density_map(points, image_shape, sigma=8):
    """
    Generate a density map by placing a Gaussian kernel at each point.

    Args:
        points: array-like of shape [N, 2], (x, y).
        image_shape: (H, W)
        sigma: Gaussian sigma (paper setting: 8)
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    if points is None:
        return density
    if isinstance(points, list):
        points = np.array(points, dtype=np.float32)
    if len(points) == 0:
        return density

    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1.0

    density = gaussian_filter(density, sigma=sigma, mode="constant")
    return density.astype(np.float32)


def generate_attention_mask(points, image_shape, radius=2):
    """
    Generate a sparse binary attention target directly from point labels.

    Args:
        points: array-like of shape [N, 2], (x, y).
        image_shape: (H, W)
        radius: local dilation radius around each point.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.float32)

    if points is None:
        return mask
    if isinstance(points, list):
        points = np.array(points, dtype=np.float32)
    if len(points) == 0:
        return mask

    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = 1.0

    radius = max(0, int(radius))
    if radius > 0:
        kernel_size = 2 * radius + 1
        mask = maximum_filter(mask, size=kernel_size, mode="constant")

    return (mask > 0).astype(np.float32)
