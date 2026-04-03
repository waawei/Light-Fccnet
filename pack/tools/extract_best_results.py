"""Batch-extract summary fields from best_model.pth checkpoints.

Usage:
    python -m pack.tools.extract_best_results --root /root/autodl-tmp/checkpoints
    python -m pack.tools.extract_best_results --root /root/autodl-tmp/checkpoints --format csv
    python -m pack.tools.extract_best_results --root /root/autodl-tmp/checkpoints --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract best checkpoint metrics from best_model.pth files")
    parser.add_argument("--root", required=True, help="Root directory containing run folders")
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format",
    )
    parser.add_argument("--output", default="", help="Optional output file path")
    return parser.parse_args()


def torch_load_compat(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_dataset(run_name: str, cfg: dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    for key, value in data_cfg.items():
        if key.endswith("_root") and isinstance(value, str) and value:
            return key[: -len("_root")]
        if key.endswith("_train_csv") and isinstance(value, str) and value:
            return key[: -len("_train_csv")]
    return run_name.split("_", 1)[0] if "_" in run_name else run_name


def infer_variant(run_name: str) -> str:
    return run_name.split("_", 1)[1] if "_" in run_name else ""


def extract_ablation_flags(cfg: dict[str, Any]) -> tuple[bool, bool, bool]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}

    ablation_mode = model_cfg.get("ablation_mode")
    legacy_use_p1 = "p1" in ablation_mode if isinstance(ablation_mode, str) else False
    legacy_use_p2 = "p2" in ablation_mode if isinstance(ablation_mode, str) else False
    legacy_use_p3 = "p3" in ablation_mode if isinstance(ablation_mode, str) else False

    use_p1 = bool(model_cfg.get("use_p1", legacy_use_p1))
    requested_use_p2 = bool(model_cfg.get("use_p2", legacy_use_p2))
    use_p2 = requested_use_p2 and use_p1
    use_p3 = bool(training_cfg.get("use_p3_loss", model_cfg.get("use_p3", legacy_use_p3)))
    return use_p1, use_p2, use_p3


def extract_one(run_dir: str) -> dict[str, Any] | None:
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        return None

    payload = torch_load_compat(ckpt_path)
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    run_name = os.path.basename(run_dir)
    use_p1, use_p2, use_p3 = extract_ablation_flags(cfg)

    return {
        "run": run_name,
        "dataset": infer_dataset(run_name, cfg),
        "variant": infer_variant(run_name),
        "epoch": payload.get("epoch"),
        "best_mae": payload.get("best_mae"),
        "best_mse": (payload.get("best_metrics") or {}).get("MSE"),
        "best_mape": (payload.get("best_metrics") or {}).get("MAPE"),
        "model_name": model_cfg.get("name", ""),
        "input_size": "x".join(str(x) for x in model_cfg.get("input_size", [])),
        "use_p1": int(use_p1),
        "use_p2": int(use_p2),
        "use_p3": int(use_p3),
        "checkpoint": ckpt_path,
    }


def collect_results(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        row = extract_one(run_dir)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda x: (str(x["dataset"]), float(x["best_mae"]) if x["best_mae"] is not None else float("inf")))
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["run", "dataset", "variant", "epoch", "best_mae", "best_mse", "best_mape", "model_name", "input_size", "use_p1", "use_p2", "use_p3"]
    widths = {h: len(h) for h in headers}
    formatted_rows: list[dict[str, str]] = []

    for row in rows:
        item = {
            "run": str(row["run"]),
            "dataset": str(row["dataset"]),
            "variant": str(row["variant"]),
            "epoch": "" if row["epoch"] is None else str(row["epoch"]),
            "best_mae": "" if row["best_mae"] is None else f"{float(row['best_mae']):.4f}",
            "best_mse": "" if row["best_mse"] is None else f"{float(row['best_mse']):.4f}",
            "best_mape": "" if row["best_mape"] is None else f"{float(row['best_mape']):.4f}",
            "model_name": str(row["model_name"]),
            "input_size": str(row["input_size"]),
            "use_p1": str(row["use_p1"]),
            "use_p2": str(row["use_p2"]),
            "use_p3": str(row["use_p3"]),
        }
        formatted_rows.append(item)
        for h, v in item.items():
            widths[h] = max(widths[h], len(v))

    lines = []
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "  ".join("-" * widths[h] for h in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for row in formatted_rows:
        lines.append("  ".join(row[h].ljust(widths[h]) for h in headers))
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    fieldnames = ["run", "dataset", "variant", "epoch", "best_mae", "best_mse", "best_mape", "model_name", "input_size", "use_p1", "use_p2", "use_p3", "checkpoint"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect_results(args.root)

    if args.format == "json":
        content = json.dumps(rows, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(content)
        return

    if args.format == "csv":
        if args.output:
            write_csv(rows, args.output)
        else:
            writer = csv.DictWriter(
                os.sys.stdout,
                fieldnames=["run", "dataset", "variant", "epoch", "best_mae", "best_mse", "best_mape", "model_name", "input_size", "use_p1", "use_p2", "use_p3", "checkpoint"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return

    content = render_table(rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
