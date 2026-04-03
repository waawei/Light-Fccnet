"""Light-FCCNet model."""
import warnings

import torch
import torch.nn as nn

from .modules import ConvBNAct, LightDensityHead, LightMultiAttentionFusion, LightPyramidFeatureAggregation


class LightFCCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        input_size=(256, 256),
        stage_channels=(32, 64, 96, 128),
        fusion_channels: int = 96,
        spatial_max_tokens: int = 1024,
        head_init_bias: float = -9.0,
        use_p1: bool = False,
        use_p2: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = tuple(input_size)
        self.use_p1 = bool(use_p1)
        requested_use_p2 = bool(use_p2)
        self.use_p2 = requested_use_p2 and self.use_p1
        if requested_use_p2 and not self.use_p1:
            warnings.warn(
                "model.use_p2=True requires model.use_p1=True; disabling use_p2 for paper-aligned semantics.",
                UserWarning,
                stacklevel=2,
            )
        self.head_init_bias = float(head_init_bias)

        self.backbone = LightPyramidFeatureAggregation(in_channels=in_channels, stage_channels=tuple(stage_channels))
        self.baseline_projection = ConvBNAct(
            self.backbone.single_scale_channels,
            fusion_channels,
            kernel_size=1,
            stride=1,
        )
        self.attention_fusion = LightMultiAttentionFusion(
            self.backbone.out_channels,
            out_channels=fusion_channels,
            spatial_max_tokens=spatial_max_tokens,
        )
        self.density_head = LightDensityHead(
            fusion_channels,
            hidden_channels=fusion_channels,
            init_bias=self.head_init_bias,
        )

    def _resize_outputs(self, x: torch.Tensor, density: torch.Tensor, attention: torch.Tensor):
        final_density = density
        if final_density.shape[2:] != x.shape[2:]:
            final_density = nn.functional.interpolate(final_density, size=x.shape[2:], mode="bilinear", align_corners=False)
            density = nn.functional.interpolate(density, size=x.shape[2:], mode="bilinear", align_corners=False)
            attention = nn.functional.interpolate(attention, size=x.shape[2:], mode="bilinear", align_corners=False)
        return final_density, attention, density

    def forward(self, x: torch.Tensor):
        if not self.use_p1:
            fused = self.baseline_projection(self.backbone.forward_single_scale(x))
            attention = fused.new_ones((fused.shape[0], 1, fused.shape[2], fused.shape[3]))
        else:
            feats = self.backbone(x)
            fused, attention = self.attention_fusion(feats, use_attention=self.use_p2)
        density = self.density_head(fused)
        return self._resize_outputs(x, density, attention)

    def predict_count(self, x: torch.Tensor):
        final_density, _, _ = self.forward(x)
        return final_density.sum(dim=(1, 2, 3))
