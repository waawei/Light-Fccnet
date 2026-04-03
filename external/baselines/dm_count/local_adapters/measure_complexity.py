"""Complexity wrapper for upstream DM-Count without pretrained-weight download."""

from __future__ import annotations

import sys
from pathlib import Path

from pack.tools.measure_model_complexity import measure_model


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_dmcount_model_for_complexity():
    from external.baselines.dm_count.upstream.models import VGG, cfg, make_layers

    return VGG(make_layers(cfg["E"]))


def measure_dmcount_complexity(input_shape=(1, 3, 1080, 1920)) -> dict[str, object]:
    model = build_dmcount_model_for_complexity()
    report = measure_model(model, input_shape)
    report["model_name"] = "dm_count"
    return report
