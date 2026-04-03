"""Dataset adapter that maps current project counting datasets to SASNet tuples."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from pack.data import GWHDDataset, MTCDataset, URCDataset
from pack.data.transforms import get_train_transforms, get_val_transforms

from .density_targets import build_sasnet_density_target


class SASNetDatasetAdapter(Dataset):
    def __init__(self, source_dataset: Dataset, split: str = "train", sigma: int = 4):
        super().__init__()
        self.source_dataset = source_dataset
        self.split = str(split).lower()
        self.sigma = int(sigma)

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected train, val, or test")

    @staticmethod
    def _require_tensor_image(sample: dict[str, Any]) -> torch.Tensor:
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError("SASNetDatasetAdapter expects source_dataset['image'] to be a torch.Tensor")
        return image.float()

    @staticmethod
    def _resolve_points(sample: dict[str, Any]) -> torch.Tensor:
        points = sample.get("points")
        if points is None:
            return torch.zeros((0, 2), dtype=torch.float32)
        if isinstance(points, torch.Tensor):
            return points.float().reshape(-1, 2)
        return torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)

    @staticmethod
    def _resolve_count(sample: dict[str, Any], points: torch.Tensor, density_target: torch.Tensor | None = None) -> float:
        count = sample.get("count")
        if count is not None:
            if isinstance(count, torch.Tensor):
                return float(count.reshape(-1)[0].item())
            return float(count)
        if density_target is not None:
            return float(density_target.sum().item())
        return float(points.shape[0])

    @staticmethod
    def _resolve_name(sample: dict[str, Any], index: int) -> str:
        image_name = sample.get("image_name")
        if image_name is None:
            return f"sample_{index}"
        return str(image_name)

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, index: int):
        sample = self.source_dataset[index]
        image = self._require_tensor_image(sample)
        points = self._resolve_points(sample)
        density_target = build_sasnet_density_target(
            density=sample.get("density"),
            points=points,
            image_shape=(int(image.shape[-2]), int(image.shape[-1])),
            sigma=self.sigma,
        )

        if self.split == "train":
            return image, density_target

        count = self._resolve_count(sample, points, density_target=density_target)
        name = self._resolve_name(sample, index)
        return image, count, name


def build_sasnet_datasets(config: dict, dataset_name: str):
    dataset_key = str(dataset_name).lower()
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    input_size = tuple(model_cfg.get("input_size", [256, 256]))
    sigma = int(data_cfg.get("sigma", 4))
    attention_radius = int(data_cfg.get("attention_radius", 2))

    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms()

    if dataset_key == "gwhd":
        train_source = GWHDDataset(
            data_cfg["gwhd_train_csv"],
            data_cfg["gwhd_images_dir"],
            transform=train_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
        val_source = GWHDDataset(
            data_cfg["gwhd_val_csv"],
            data_cfg["gwhd_images_dir"],
            transform=val_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
    elif dataset_key == "mtc":
        train_source = MTCDataset(
            data_cfg["mtc_root"],
            split="train",
            split_file=data_cfg.get("mtc_train_split_file"),
            transform=train_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
        val_source = MTCDataset(
            data_cfg["mtc_root"],
            split="val",
            split_file=data_cfg.get("mtc_val_split_file"),
            transform=val_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
    elif dataset_key == "urc":
        train_source = URCDataset(
            data_cfg["urc_root"],
            split="train",
            split_file=data_cfg.get("urc_train_split_file"),
            transform=train_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
        val_source = URCDataset(
            data_cfg["urc_root"],
            split="val",
            split_file=data_cfg.get("urc_val_split_file"),
            transform=val_transform,
            target_size=input_size,
            sigma=sigma,
            attention_radius=attention_radius,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Expected gwhd, mtc, or urc")

    return {
        "train": SASNetDatasetAdapter(train_source, split="train", sigma=sigma),
        "val": SASNetDatasetAdapter(val_source, split="val", sigma=sigma),
    }

