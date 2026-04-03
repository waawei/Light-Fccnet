"""Measure parameter count and approximate FLOPs for local models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


TensorShape = Tuple[int, int, int, int]
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_input_shape(values: Iterable[str]) -> TensorShape:
    shape = tuple(int(v) for v in values)
    if len(shape) != 4:
        raise ValueError("input shape must contain exactly 4 integers: N C H W")
    return shape  # type: ignore[return-value]


def load_config(path: Path | str) -> Dict:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _first_tensor(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class FlopCounter:
    """Approximate FLOP counter using forward hooks on common layers."""

    def __init__(self) -> None:
        self.total_flops = 0
        self._handles = []

    def _conv2d_flops(self, module: nn.Conv2d, output: torch.Tensor) -> int:
        out = output.shape
        batch, out_channels, out_h, out_w = out
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        filters_per_channel = out_channels // groups
        conv_per_position = kernel_h * kernel_w * in_channels * filters_per_channel // groups
        bias_ops = 1 if module.bias is not None else 0
        return int(batch * out_h * out_w * (2 * conv_per_position + bias_ops))

    @staticmethod
    def _linear_flops(module: nn.Linear, output: torch.Tensor) -> int:
        batch = output.shape[0] if output.ndim > 1 else 1
        bias_ops = 1 if module.bias is not None else 0
        return int(batch * module.out_features * (2 * module.in_features + bias_ops))

    @staticmethod
    def _batchnorm_flops(output: torch.Tensor) -> int:
        return int(output.numel() * 2)

    @staticmethod
    def _activation_flops(output: torch.Tensor) -> int:
        return int(output.numel())

    @staticmethod
    def _pool_flops(output: torch.Tensor) -> int:
        return int(output.numel())

    @staticmethod
    def _upsample_flops(output: torch.Tensor, mode: str | None) -> int:
        if mode == "bilinear":
            return int(output.numel() * 4)
        return int(output.numel())

    def _hook(self, module: nn.Module, inputs, output) -> None:
        tensor = _first_tensor(output)
        if tensor is None:
            return

        if isinstance(module, nn.Conv2d):
            self.total_flops += self._conv2d_flops(module, tensor)
        elif isinstance(module, nn.Linear):
            self.total_flops += self._linear_flops(module, tensor)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self.total_flops += self._batchnorm_flops(tensor)
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.LeakyReLU, nn.GELU)):
            self.total_flops += self._activation_flops(tensor)
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            self.total_flops += self._pool_flops(tensor)
        elif isinstance(module, nn.Upsample):
            self.total_flops += self._upsample_flops(tensor, getattr(module, "mode", None))

    def install(self, model: nn.Module) -> None:
        import torch.nn as nn

        supported = (
            nn.Conv2d,
            nn.Linear,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.ReLU,
            nn.ReLU6,
            nn.Sigmoid,
            nn.LeakyReLU,
            nn.GELU,
            nn.MaxPool2d,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveMaxPool2d,
            nn.Upsample,
        )
        for module in model.modules():
            if isinstance(module, supported):
                self._handles.append(module.register_forward_hook(self._hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def measure_model(model: nn.Module, input_shape: TensorShape) -> Dict[str, int | TensorShape]:
    import torch

    params = sum(p.numel() for p in model.parameters())
    counter = FlopCounter()
    counter.install(model)

    model.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        _ = model(dummy)

    counter.remove()
    return {
        "params": int(params),
        "flops": int(counter.total_flops),
        "input_shape": tuple(int(v) for v in input_shape),
    }


def measure_from_config(path: Path | str, input_shape: TensorShape | None = None) -> Dict[str, int | str | TensorShape]:
    from pack.models import build_model

    config = load_config(path)
    model = build_model(config)
    model_name = config.get("model", {}).get("name", model.__class__.__name__.lower())

    if input_shape is None:
        input_size = tuple(config.get("model", {}).get("input_size", [256, 256]))
        input_shape = (1, config.get("model", {}).get("in_channels", 3), input_size[0], input_size[1])

    report = measure_model(model, input_shape)
    report["model_name"] = model_name
    report["config_path"] = str(path)
    return report


def format_human_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure params and approximate FLOPs for a model config.")
    parser.add_argument("--config", required=True, type=str, help="Path to a YAML config file.")
    parser.add_argument(
        "--input-shape",
        nargs=4,
        metavar=("N", "C", "H", "W"),
        default=None,
        help="Optional input tensor shape. Defaults to config-driven shape.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    input_shape = parse_input_shape(args.input_shape) if args.input_shape is not None else None
    report = measure_from_config(args.config, input_shape=input_shape)

    print(f"Model: {report['model_name']}")
    print(f"Config: {report['config_path']}")
    print(f"Input shape: {report['input_shape']}")
    print(f"Params: {report['params']} ({format_human_count(int(report['params']))})")
    print(f"Approx FLOPs: {report['flops']} ({format_human_count(int(report['flops']))})")


if __name__ == "__main__":
    main()
