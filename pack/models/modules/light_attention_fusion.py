"""Multi-attention fusion for Light-FCCNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .light_channel_attention import LightChannelAttention
from .light_spatial_attention import LightSpatialAttention


class LightMultiAttentionFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels: int, spatial_max_tokens: int = 1024):
        super().__init__()
        self.align = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for in_channels in in_channels_list
            ]
        )
        att_hidden_channels = max(out_channels // 2, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_attention = LightSpatialAttention(out_channels, max_tokens=spatial_max_tokens)
        self.channel_attention = LightChannelAttention(out_channels, max_tokens=spatial_max_tokens)
        self.attention_head = nn.Sequential(
            nn.Conv2d(out_channels, att_hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(att_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(att_hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _align_features(self, feats):
        target_size = feats[0].shape[2:]
        aligned = []
        for feat, proj in zip(feats, self.align):
            x = proj(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(x)
        return self.fuse(torch.cat(aligned, dim=1))

    def _bypass_attention(self, fused: torch.Tensor):
        attention = fused.new_ones((fused.shape[0], 1, fused.shape[2], fused.shape[3]))
        return fused, attention

    def forward(self, feats, use_attention: bool = True):
        fused = self._align_features(feats)
        if not use_attention:
            return self._bypass_attention(fused)
        fused = self.spatial_attention(fused)
        fused = self.channel_attention(fused)
        attention = self.attention_head(fused)
        return fused, attention
