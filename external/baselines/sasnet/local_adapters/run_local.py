"""Minimal local CLI for running SASNet adaptation on current project datasets."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from .eval import build_result_row
from .export_results import append_result_row_csv, save_result_row_json
from .runner import evaluate_sasnet_model
from .train_bridge import build_sasnet_dataloaders, run_train_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local SASNet adaptation on current project datasets.")
    parser.add_argument("--config", required=True, type=str, help="Path to a pack YAML config.")
    parser.add_argument("--dataset-name", required=True, choices=["gwhd", "mtc", "urc"], help="Target dataset name.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for this local adapter run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional num_workers override.")
    parser.add_argument("--save-dir", type=str, default="external/baselines/sasnet/runs")
    parser.add_argument("--save-prefix", type=str, default="sasnet_local")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint to load before evaluation.")
    parser.add_argument("--block-size", type=int, default=32, help="Patch size for feature-level selection.")
    parser.add_argument("--log-para", type=float, default=1000.0, help="Density magnification used by upstream SASNet.")
    return parser


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_sasnet_model(block_size: int = 32):
    from external.baselines.sasnet.upstream.model import SASNet

    args = Namespace(block_size=int(block_size))
    return SASNet(pretrained=False, args=args)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    train_cfg = config.setdefault("training", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)

    device = torch.device(args.device)
    dataloaders = build_sasnet_dataloaders(config, dataset_name=args.dataset_name)
    model = build_sasnet_model(block_size=args.block_size).to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", train_cfg.get("optimizer", {}).get("weight_decay", 1e-4))),
        )
        criterion = torch.nn.MSELoss().to(device)

        for epoch in range(1, int(args.epochs) + 1):
            for batch in dataloaders["train"]:
                run_train_batch(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    log_para=float(args.log_para),
                )
            epoch_path = save_dir / f"{args.save_prefix}_{args.dataset_name}_epoch{epoch}.pth"
            torch.save(model.state_dict(), epoch_path)

    report = evaluate_sasnet_model(
        model,
        dataloaders["val"],
        device=device,
        dataset_name=args.dataset_name.upper(),
        result_type="adapted reproduction",
        log_para=float(args.log_para),
    )
    summary = build_result_row(
        dataset=args.dataset_name.upper(),
        metrics=report["metrics"],
        result_type="adapted reproduction",
        checkpoint_path=args.checkpoint or "",
    )

    json_path = save_dir / f"{args.save_prefix}_{args.dataset_name}_summary.json"
    csv_path = save_dir / f"{args.save_prefix}_{args.dataset_name}_summary.csv"
    save_result_row_json(summary, json_path)
    append_result_row_csv(summary, csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
