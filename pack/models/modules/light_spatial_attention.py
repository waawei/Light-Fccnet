"""Spatial attention for Light-FCCNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightSpatialAttention(nn.Module):
    def __init__(self, channels: int, max_tokens: int = 1024):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pooled_size(self, h: int, w: int) -> tuple[int, int]:
        if self.max_tokens <= 0 or h * w <= self.max_tokens:
            return h, w

        scale = (float(self.max_tokens) / float(h * w)) ** 0.5
        pooled_h = max(1, min(h, int(h * scale)))
        pooled_w = max(1, min(w, int(w * scale)))

        while pooled_h * pooled_w > self.max_tokens:
            if pooled_h >= pooled_w and pooled_h > 1:
                pooled_h -= 1
            elif pooled_w > 1:
                pooled_w -= 1
            else:
                break

        return pooled_h, pooled_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pooled_h, pooled_w = self._pooled_size(h, w)
        source = x
        if (pooled_h, pooled_w) != (h, w):
            source = F.adaptive_avg_pool2d(x, output_size=(pooled_h, pooled_w))

        n = pooled_h * pooled_w
        query = self.query(source).reshape(b, -1, n).permute(0, 2, 1)
        key = self.key(source).reshape(b, -1, n)
        att = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(source).reshape(b, c, n)
        context = torch.bmm(value, att.permute(0, 2, 1)).reshape(b, c, pooled_h, pooled_w)
        if (pooled_h, pooled_w) != (h, w):
            context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)
        return x + self.gamma * context
