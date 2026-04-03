"""Helpers for optional density-target scaling."""


def scale_density_target(target, density_scale: float = 1.0):
    density_scale = float(density_scale)
    if density_scale == 1.0:
        return target
    return target * density_scale


def descale_count(count, density_scale: float = 1.0):
    density_scale = float(density_scale)
    if density_scale == 1.0:
        return count
    return count / density_scale
