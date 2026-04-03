"""GWHD dataset loader for density-map counting."""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .point_supervision import apply_transform_with_points, build_point_supervision

if hasattr(Image, "Resampling"):
    _BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
else:
    _BILINEAR_RESAMPLE = getattr(Image, "BILINEAR", 2)


class GWHDDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, target_size=(256, 256), sigma=8, attention_radius=2):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.attention_radius = attention_radius

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _parse_boxes(boxes_string):
        if pd.isna(boxes_string) or not str(boxes_string).strip():
            return []
        if str(boxes_string).strip().lower() == "no_box":
            return []

        boxes = []
        for box_str in str(boxes_string).split(";"):
            vals = box_str.strip().split()
            if len(vals) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, vals)
                boxes.append([x1, y1, x2, y2])
            except ValueError:
                continue
        return boxes

    @staticmethod
    def _boxes_to_points(boxes):
        points = []
        for x1, y1, x2, y2 in boxes:
            points.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        return points

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["image_name"]
        image_path = os.path.join(self.images_dir, image_name)

        image = np.array(Image.open(image_path).convert("RGB"))
        oh, ow = image.shape[:2]
        th, tw = self.target_size

        boxes = self._parse_boxes(row["BoxesString"])
        points = self._boxes_to_points(boxes)
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
            "image_name": image_name,
        }
