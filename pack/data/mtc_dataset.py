"""MTC dataset loader for density-map counting."""
import json
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from .point_supervision import apply_transform_with_points, build_point_supervision

if hasattr(Image, "Resampling"):
    _BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
else:
    _BILINEAR_RESAMPLE = getattr(Image, "BILINEAR", 2)


class MTCDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        target_size=(256, 256),
        sigma=8,
        attention_radius=2,
        split="train",
        split_file=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.attention_radius = attention_radius
        self.split = split
        self.split_file = split_file
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        split_file = self.split_file if self.split_file is not None else os.path.join(self.root_dir, f"{self.split}.txt")
        if os.path.exists(split_file):
            self._collect_from_split_file(split_file)
        else:
            self._collect_standard_layout()

    def _collect_from_split_file(self, split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # Standard split: image_path ann_path
                    img_path = os.path.join(self.root_dir, parts[0].lstrip("/\\"))
                    ann_path = os.path.join(self.root_dir, parts[1].lstrip("/\\"))
                    if os.path.exists(img_path) and os.path.exists(ann_path):
                        fmt = "uav_csv" if ann_path.lower().endswith(".csv") else "mat"
                        self.samples.append(
                            {
                                "image_path": img_path,
                                "ann_path": ann_path,
                                "format": fmt,
                                "image_name": os.path.basename(img_path),
                            }
                        )

    def _collect_standard_layout(self):
        for sub in os.listdir(self.root_dir):
            sub_dir = os.path.join(self.root_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            images_dir = os.path.join(sub_dir, "Images")
            ann_dir = os.path.join(sub_dir, "Annotations")
            if not (os.path.exists(images_dir) and os.path.exists(ann_dir)):
                continue

            for fname in os.listdir(images_dir):
                if not fname.lower().endswith(".jpg"):
                    continue
                img_path = os.path.join(images_dir, fname)
                ann_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + ".mat")
                if os.path.exists(ann_path):
                    self.samples.append(
                        {"image_path": img_path, "ann_path": ann_path, "format": "mat", "image_name": fname}
                    )

    @staticmethod
    def _load_csv_points(csv_path):
        points = []
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                attrs = json.loads(row["region_shape_attributes"])
                points.append([float(attrs.get("cx", 0)), float(attrs.get("cy", 0))])
        except Exception:
            pass
        return points

    @staticmethod
    def _load_mat_points(mat_path):
        points = []
        try:
            mat = loadmat(mat_path)
            if "annotation" in mat:
                ann = mat["annotation"]
                if isinstance(ann, np.ndarray) and ann.size > 0:
                    cell = ann[0, 0]
                    if isinstance(cell, np.void) and cell.dtype.names and "bndbox" in cell.dtype.names:
                        bndbox = cell["bndbox"]
                        if isinstance(bndbox, np.ndarray) and bndbox.size > 0:
                            points = bndbox.tolist()
            elif "points" in mat:
                points = np.array(mat["points"]).reshape(-1, 2).tolist()
            elif "location" in mat:
                points = np.array(mat["location"]).reshape(-1, 2).tolist()
        except Exception:
            pass
        return points

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        oh, ow = image.shape[:2]
        th, tw = self.target_size

        if sample["format"] == "uav_csv":
            points = self._load_csv_points(sample["ann_path"])
        else:
            points = self._load_mat_points(sample["ann_path"])

        if len(points) > 0:
            points = np.array(points, dtype=np.float32)
            points[:, 0] *= tw / float(ow)
            points[:, 1] *= th / float(oh)
        else:
            points = np.zeros((0, 2), dtype=np.float32)

        image = np.array(Image.fromarray(image).resize((tw, th), _BILINEAR_RESAMPLE))
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
