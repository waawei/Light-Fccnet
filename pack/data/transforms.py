"""Data augmentation aligned to manuscript description."""
from typing import Any, Sequence, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _compose(transforms: Sequence[Any]) -> A.Compose:
    # Pylance/pyright compatibility: Compose's generic expects its own
    # internal transform type list; cast keeps runtime behavior unchanged.
    return A.Compose(
        cast(list[Any], list(transforms)),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def _int_range(lo: Any, hi: Any, lower_bound: int = 1) -> tuple[int, int]:
    lo_i = max(lower_bound, int(lo))
    hi_i = max(lo_i, int(hi))
    return lo_i, hi_i


def _build_coarse_dropout(config: dict, h: int, w: int):
    data_cfg = config.get("data", {})
    if not bool(data_cfg.get("use_coarse_dropout", False)):
        return None

    p = float(data_cfg.get("coarse_dropout_p", 1.0))
    p = min(max(p, 0.0), 1.0)

    min_holes, max_holes = _int_range(
        data_cfg.get("coarse_dropout_min_holes", 1),
        data_cfg.get("coarse_dropout_max_holes", 6),
        lower_bound=1,
    )
    min_h, max_h = _int_range(
        data_cfg.get("coarse_dropout_min_height", max(8, h // 16)),
        data_cfg.get("coarse_dropout_max_height", max(16, h // 6)),
        lower_bound=1,
    )
    min_w, max_w = _int_range(
        data_cfg.get("coarse_dropout_min_width", max(8, w // 16)),
        data_cfg.get("coarse_dropout_max_width", max(16, w // 6)),
        lower_bound=1,
    )
    fill_value = int(data_cfg.get("coarse_dropout_fill_value", 0))

    # Keep compatibility with different albumentations versions.
    coarse_dropout_ctor = cast(Any, A.CoarseDropout)
    try:
        return coarse_dropout_ctor(
            num_holes_range=(min_holes, max_holes),
            hole_height_range=(min_h, max_h),
            hole_width_range=(min_w, max_w),
            fill=fill_value,
            p=p,
        )
    except TypeError:
        legacy_kwargs: dict[str, Any] = {
            "max_holes": max_holes,
            "min_holes": min_holes,
            "max_height": max_h,
            "min_height": min_h,
            "max_width": max_w,
            "min_width": min_w,
            "fill_value": fill_value,
            "p": p,
        }
        return coarse_dropout_ctor(**legacy_kwargs)


def get_train_transforms(config):
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    input_size = model_cfg.get("input_size", [256, 256])
    h, w = input_size[0], input_size[1]

    # Paper-described augmentation:
    # - random horizontal flipping
    # - random cropping
    # - color jittering
    transforms_list: list[Any] = []
    if bool(data_cfg.get("use_random_resized_crop", True)):
        transforms_list.append(
            A.RandomResizedCrop(
                size=(h, w),
                scale=tuple(data_cfg.get("random_resized_crop_scale", (0.8, 1.0))),
                ratio=tuple(data_cfg.get("random_resized_crop_ratio", (0.9, 1.1))),
                p=float(data_cfg.get("random_resized_crop_p", 0.5)),
            )
        )

    horizontal_flip_p = float(data_cfg.get("horizontal_flip_p", 0.5))
    if horizontal_flip_p > 0.0:
        transforms_list.append(A.HorizontalFlip(p=horizontal_flip_p))

    if bool(data_cfg.get("use_color_jitter", True)):
        transforms_list.append(
            A.ColorJitter(
                brightness=float(data_cfg.get("color_jitter_brightness", 0.2)),
                contrast=float(data_cfg.get("color_jitter_contrast", 0.2)),
                saturation=float(data_cfg.get("color_jitter_saturation", 0.2)),
                hue=float(data_cfg.get("color_jitter_hue", 0.1)),
                p=float(data_cfg.get("color_jitter_p", 0.5)),
            )
        )

    coarse_dropout = _build_coarse_dropout(config, h, w)
    if coarse_dropout is not None:
        transforms_list.append(coarse_dropout)
    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return _compose(transforms_list)


def get_val_transforms():
    val_transforms: Sequence[object] = (
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    )
    return _compose(val_transforms)
