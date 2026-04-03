"""Local CSRNet baseline adapted to the Light-FCCNet training contract."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_frontend(
    in_channels: int,
    conv_channels: tuple[int, ...],
    pool_indices: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = int(in_channels)
    pool_index_set = set(int(index) for index in pool_indices)

    for index, out_channels in enumerate(conv_channels):
        layers.extend(
            [
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        )
        if index in pool_index_set:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        current_channels = int(out_channels)

    return nn.Sequential(*layers)


def _make_backend(in_channels: int, conv_channels: tuple[int, ...], dilation: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = int(in_channels)

    for out_channels in conv_channels:
        layers.extend(
            [
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
            ]
        )
        current_channels = int(out_channels)

    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size: tuple[int, int] = (256, 256),
        frontend_channels: tuple[int, ...] = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512),
        frontend_pool_indices: tuple[int, ...] = (1, 3, 6),
        backend_channels: tuple[int, ...] = (512, 512, 512, 256, 128, 64),
        dilation: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_size = tuple(int(v) for v in input_size)
        self.use_p1 = False
        self.use_p2 = False

        self.frontend = _make_frontend(
            in_channels=self.in_channels,
            conv_channels=tuple(int(v) for v in frontend_channels),
            pool_indices=tuple(int(v) for v in frontend_pool_indices),
        )
        frontend_out = int(frontend_channels[-1])
        self.backend = _make_backend(
            in_channels=frontend_out,
            conv_channels=tuple(int(v) for v in backend_channels),
            dilation=int(dilation),
        )
        self.output_layer = nn.Conv2d(int(backend_channels[-1]), 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _resize_to_input(self, x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        if density.shape[2:] == x.shape[2:]:
            return density
        return F.interpolate(density, size=x.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        features = self.frontend(x)
        features = self.backend(features)
        density = self.output_layer(features)
        density = self._resize_to_input(x, density)
        attention = density.new_ones((density.shape[0], 1, density.shape[2], density.shape[3]))
        return density, attention, density

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        final_density, _, _ = self.forward(x)
        return final_density.sum(dim=(1, 2, 3))
