"""Complexity wrapper for upstream SASNet without pretrained-weight download."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from pack.tools.measure_model_complexity import measure_model


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_sasnet_model_for_complexity(block_size: int = 32):
    from external.baselines.sasnet.upstream.model import SASNet

    args = Namespace(block_size=int(block_size))
    return SASNet(pretrained=False, args=args)


def measure_sasnet_complexity(input_shape=(1, 3, 1080, 1920), block_size: int = 32) -> dict[str, object]:
    model = build_sasnet_model_for_complexity(block_size=block_size)
    report = measure_model(model, input_shape)
    report["model_name"] = "sasnet"
    return report
