"""URC dataset loader for density-map counting."""
import os
from typing import Optional
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .point_supervision import apply_transform_with_points, build_point_supervision

if hasattr(Image, "Resampling"):
    _BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
else:
    _BILINEAR_RESAMPLE = getattr(Image, "BILINEAR", 2)


class URCDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        split_file: Optional[str] = None,
        transform=None,
        target_size=(256, 256),
        sigma=8,
        attention_radius=2,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_file = split_file
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.attention_radius = attention_radius
        self.samples = self._collect_samples()

    def _collect_samples(self):
        if self.split_file is not None and os.path.exists(self.split_file):
            return self._collect_from_split_file(self.split_file)

        split_dir = os.path.join(self.root_dir, self.split)
        img_dir = os.path.join(split_dir, "imgs_4")
        if self.split == "train":
            ann_candidates = [os.path.join(split_dir, "new_data_4"), os.path.join(split_dir, "dis_data_4")]
        else:
            ann_candidates = [os.path.join(split_dir, "dis_data_4")]

        ann_dir = None
        for d in ann_candidates:
            if os.path.exists(d):
                ann_dir = d
                break

        if not os.path.exists(img_dir) or ann_dir is None:
            return []

        samples = []
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.endswith(".jpg"):
                continue
            img_path = os.path.join(img_dir, img_file)
            ann_path = os.path.join(ann_dir, os.path.splitext(img_file)[0] + ".h5")
            if os.path.exists(ann_path):
                samples.append({"image_path": img_path, "ann_path": ann_path, "image_name": img_file})
        return samples

    def _collect_from_split_file(self, split_file):
        samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_rel, ann_rel = parts[0], parts[1]
                img_path = os.path.join(self.root_dir, img_rel.lstrip("/\\"))
                ann_path = os.path.join(self.root_dir, ann_rel.lstrip("/\\"))
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    samples.append(
                        {
                            "image_path": img_path,
                            "ann_path": ann_path,
                            "image_name": os.path.basename(img_path),
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize_points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1 and points.size > 0:
            points = points.reshape(-1, 2)
        elif points.ndim == 3:
            points = points[0]
        if points.ndim != 2 or (points.shape[1] if points.ndim == 2 else 0) != 2:
            return np.zeros((0, 2), dtype=np.float32)
        return points

    @staticmethod
    def _extract_points_from_kpoint(kpoint_obj):
        if not isinstance(kpoint_obj, h5py.Dataset):
            return np.zeros((0, 2), dtype=np.float32)
        kpoint = np.asarray(kpoint_obj[()])
        if kpoint.ndim != 2:
            return np.zeros((0, 2), dtype=np.float32)
        ys, xs = np.nonzero(kpoint > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    @staticmethod
    def _extract_point_array(h5_file):
        coord_obj = h5_file.get("coordinate")
        points_obj = h5_file.get("points")

        if isinstance(coord_obj, h5py.Dataset):
            return URCDataset._normalize_points(coord_obj[()])
        if isinstance(points_obj, h5py.Dataset):
            return URCDataset._normalize_points(points_obj[()])

        for key in h5_file.keys():
            obj = h5_file[key]
            if not isinstance(obj, h5py.Dataset):
                continue
            cand = np.asarray(obj[()])
            if cand.ndim >= 2 and cand.shape[-1] == 2:
                return URCDataset._normalize_points(cand)

        return URCDataset._extract_points_from_kpoint(h5_file.get("kpoint"))

    @staticmethod
    def _extract_semantic_count(h5_file, points):
        # URC's "dis" field is not plant count; it belongs to another label
        # semantic and must not be mixed into point-count regression.
        gt_obj = h5_file.get("gt")
        if isinstance(gt_obj, h5py.Dataset):
            try:
                return int(np.asarray(gt_obj[()]).reshape(-1)[0])
            except Exception:
                pass

        density_obj = h5_file.get("density")
        if isinstance(density_obj, h5py.Dataset):
            try:
                return int(round(float(np.asarray(density_obj[()]).sum())))
            except Exception:
                pass

        return int(len(points))

    @staticmethod
    def _load_points_and_count(h5_path):
        with h5py.File(h5_path, "r") as f:
            points = URCDataset._extract_point_array(f)
            count = URCDataset._extract_semantic_count(f, points)
        return points, count

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        ow, oh = image.size
        th, tw = self.target_size

        points, count = self._load_points_and_count(sample["ann_path"])
        if len(points) > 0:
            points = points.copy()
            points[:, 0] *= tw / float(ow)
            points[:, 1] *= th / float(oh)
        else:
            points = np.zeros((0, 2), dtype=np.float32)

        image = np.array(image.resize((tw, th), _BILINEAR_RESAMPLE))
        image, points = apply_transform_with_points(image, points, self.transform)
        density, att_mask, points_tensor, count_tensor = build_point_supervision(
            points,
            image_shape=(th, tw),
            sigma=self.sigma,
            attention_radius=self.attention_radius,
        )
        return {
            "image": image,
            "density": density,
            "attention_mask": att_mask,
            "count": count_tensor,
            "points": points_tensor,
            "image_name": sample["image_name"],
        }
