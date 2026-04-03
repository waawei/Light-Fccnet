"""Lightweight convolution blocks for Light-FCCNet."""
import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableDownsample(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class LightweightConvBlock(nn.Module):
    """
    Paper-aligned lightweight block:
    split F -> F1/F2
    F1 -> pointwise -> F3
    F3 -> depthwise -> F4
    F3 + F2 -> F5
    fuse(F4, F5) -> output
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels < 2:
            raise ValueError("LightweightConvBlock requires channels >= 2")
        split_channels = channels // 2
        remain_channels = channels - split_channels
        self.split_channels = split_channels
        self.remain_channels = remain_channels

        self.pointwise = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                split_channels,
                split_channels,
                kernel_size=3,
                padding=1,
                groups=split_channels,
                bias=False,
            ),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2 = torch.split(x, [self.split_channels, self.remain_channels], dim=1)
        f3 = self.pointwise(f1)
        f4 = self.depthwise(f3)

        if f2.shape[1] != f3.shape[1]:
            if f2.shape[1] > f3.shape[1]:
                f2 = f2[:, : f3.shape[1]]
            else:
                pad = f3.shape[1] - f2.shape[1]
                f2 = torch.cat([f2, f2[:, :pad]], dim=1)
        f5 = f3 + f2
        fused = torch.cat([f4, f5], dim=1)
        return self.fuse(fused)
