"""Density head for Light-FCCNet."""
import torch
import torch.nn as nn


class LightDensityHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, init_bias: float = -9.0):
        super().__init__()
        mid_channels = max(hidden_channels // 2, 1)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        # Keep density non-negative while starting from a very small baseline.
        self.out_act = nn.Softplus(beta=1.0, threshold=20.0)

        # A tiny non-zero initialization keeps the initial prediction small
        # without blocking gradients from flowing back into fusion/backbone layers.
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out_conv.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.out_act(self.out_conv(x))
        return x
