"""Bridge utilities for running SASNet with local adapters."""

from __future__ import annotations

import torch

from .eval import build_result_row, compute_count_metrics


def sasnet_train_collate(batch):
    images, density_maps = zip(*batch)
    return torch.stack(list(images), 0), torch.stack(list(density_maps), 0)


@torch.no_grad()
def evaluate_sasnet_model(
    model,
    dataloader,
    device="cpu",
    dataset_name: str = "unknown",
    result_type: str = "adapted reproduction",
    log_para: float = 1000.0,
):
    model.eval()
    model.to(device)

    pred_counts: list[float] = []
    gt_counts: list[float] = []
    rows: list[dict[str, object]] = []
    count_scale = float(log_para) if float(log_para) != 0.0 else 1.0

    for image, count, name in dataloader:
        image = image.to(device)
        outputs = model(image)
        pred_count = float(torch.sum(outputs).item()) / count_scale

        if isinstance(count, torch.Tensor):
            gt_count = float(count.reshape(-1)[0].item())
        else:
            gt_count = float(count[0] if isinstance(count, (list, tuple)) else count)

        sample_name = name[0] if isinstance(name, (list, tuple)) else name

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        rows.append(
            {
                "image_name": str(sample_name),
                "pred_count": pred_count,
                "gt_count": gt_count,
            }
        )

    metrics = compute_count_metrics(pred_counts, gt_counts)
    summary = build_result_row(
        dataset=dataset_name,
        metrics=metrics,
        result_type=result_type,
    )
    return {
        "metrics": metrics,
        "rows": rows,
        "summary": summary,
    }
