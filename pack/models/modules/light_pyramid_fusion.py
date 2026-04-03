"""Lightweight pyramid feature aggregation for Light-FCCNet."""
import torch
import torch.nn as nn

from .lightweight_conv import ConvBNAct, DepthwiseSeparableDownsample, LightweightConvBlock


class PyramidBranchStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pre_blocks: int,
        post_blocks: int,
        num_downsamples: int,
    ):
        super().__init__()
        self.input_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)
        self.pre = nn.Sequential(*[LightweightConvBlock(out_channels) for _ in range(pre_blocks)])
        self.downsample = nn.Sequential(*[DepthwiseSeparableDownsample(out_channels) for _ in range(num_downsamples)])
        self.post = nn.Sequential(*[LightweightConvBlock(out_channels) for _ in range(post_blocks)])
        self.residual = nn.Sequential(*[DepthwiseSeparableDownsample(out_channels) for _ in range(num_downsamples)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        feat = self.pre(projected)
        feat = self.downsample(feat)
        feat = self.post(feat)
        residual = self.residual(projected)
        return feat + residual


class LightPyramidFeatureAggregation(nn.Module):
    """
    Simulated manuscript-aligned pyramid:
    - Stage1 keeps full resolution and uses two lightweight blocks
    - Stage2 targets 1/4 of input
    - Stage3 targets 1/16 of input
    - Stage4 targets 1/64 of input
    """

    def __init__(self, in_channels: int = 3, stage_channels=(32, 64, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = stage_channels
        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=1)
        self.stage1 = nn.Sequential(
            LightweightConvBlock(c1),
            LightweightConvBlock(c1),
        )
        self.stage2 = PyramidBranchStage(c1, c2, pre_blocks=1, post_blocks=1, num_downsamples=2)
        self.stage3 = PyramidBranchStage(c2, c3, pre_blocks=2, post_blocks=1, num_downsamples=2)
        self.stage4 = PyramidBranchStage(c3, c4, pre_blocks=2, post_blocks=1, num_downsamples=2)
        self.out_channels = tuple(stage_channels)
        self.single_scale_channels = c1
        self.target_scales = (1, 4, 16, 64)

    def forward_single_scale(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.stage1(x)

    def forward(self, x: torch.Tensor):
        f1 = self.forward_single_scale(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
