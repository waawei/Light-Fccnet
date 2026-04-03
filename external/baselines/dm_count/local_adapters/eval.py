"""Unified metric/export helpers for local DM-Count adaptation."""

from __future__ import annotations

from typing import Iterable

from pack.utils.metrics import cal_mae, cal_mape, cal_mse


def compute_count_metrics(pred_counts: Iterable[float], gt_counts: Iterable[float]) -> dict[str, float]:
    pred = list(pred_counts)
    gt = list(gt_counts)
    return {
        "mae": float(cal_mae(pred, gt)),
        "mse": float(cal_mse(pred, gt)),
        "mape": float(cal_mape(pred, gt)),
    }


def build_result_row(
    dataset: str,
    metrics: dict[str, float],
    result_type: str,
    params: int | None = None,
    flops: int | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, object]:
    return {
        "method": "DM-Count",
        "dataset": str(dataset),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "mape": float(metrics["mape"]),
        "params": params,
        "flops": flops,
        "result_type": str(result_type),
        "checkpoint_path": checkpoint_path,
    }
