"""Channel attention for Light-FCCNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightChannelAttention(nn.Module):
    def __init__(self, channels: int, max_tokens: int = 1024):
        super().__init__()
        reduced_channels = max(channels // 2, 1)
        self.max_tokens = int(max_tokens)
        self.reduced_channels = reduced_channels
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.beta = nn.Parameter(torch.zeros(1))

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
        reduced = self.reduce(x)
        b, c_half, h, w = reduced.shape

        pooled_h, pooled_w = self._pooled_size(h, w)
        reduced_source = reduced
        if (pooled_h, pooled_w) != (h, w):
            reduced_source = F.adaptive_avg_pool2d(reduced, output_size=(pooled_h, pooled_w))

        n_small = pooled_h * pooled_w
        p = reduced_source.reshape(b, c_half, n_small).transpose(1, 2)
        q = reduced_source.reshape(b, c_half, n_small)
        w_att = torch.softmax(torch.bmm(p, q), dim=-1)
        reduced_refined = torch.bmm(w_att, p).transpose(1, 2).reshape(b, c_half, pooled_h, pooled_w)
        if (pooled_h, pooled_w) != (h, w):
            reduced_refined = F.interpolate(reduced_refined, size=(h, w), mode="bilinear", align_corners=False)
        refined = self.fusion(reduced_refined)

        b, c, h, w = x.shape
        n = h * w
        a = x.reshape(b, c, n)
        channel_att = torch.softmax(torch.bmm(a, a.transpose(1, 2)), dim=-1)
        channel_context = torch.bmm(channel_att, a).reshape(b, c, h, w)

        return refined + x + self.beta * channel_context
