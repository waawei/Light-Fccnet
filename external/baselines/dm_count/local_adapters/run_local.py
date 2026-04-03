"""Minimal local CLI for running DM-Count adaptation on current datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from external.baselines.dm_count.upstream.losses.ot_loss import OT_Loss
from external.baselines.dm_count.upstream.models import VGG, cfg, make_layers

from .eval import build_result_row
from .export_results import append_result_row_csv, save_result_row_json
from .runner import evaluate_dmcount_model
from .train_bridge import build_dmcount_dataloaders, run_train_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local DM-Count adaptation on current project datasets.")
    parser.add_argument("--config", required=True, type=str, help="Path to a pack YAML config.")
    parser.add_argument("--dataset-name", required=True, choices=["gwhd", "mtc", "urc"], help="Target dataset name.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for this local adapter run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional num_workers override.")
    parser.add_argument("--save-dir", type=str, default="external/baselines/dm_count/runs")
    parser.add_argument("--save-prefix", type=str, default="dm_count_local")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint to load before evaluation.")
    return parser


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dmcount_model(load_pretrained: bool = False):
    if load_pretrained:
        from external.baselines.dm_count.upstream.models import vgg19

        return vgg19()
    return VGG(make_layers(cfg["E"]))


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    train_cfg = config.setdefault("training", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)

    device = torch.device(args.device)
    dataloaders = build_dmcount_dataloaders(config, dataset_name=args.dataset_name)
    model = build_dmcount_model(load_pretrained=False).to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        input_size = int(config.get("model", {}).get("input_size", [256, 256])[0])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", train_cfg.get("optimizer", {}).get("weight_decay", 1e-4))),
        )
        ot_loss = OT_Loss(
            c_size=input_size,
            stride=8,
            norm_cood=int(train_cfg.get("norm_cood", 0)),
            device=device,
            num_of_iter_in_ot=int(train_cfg.get("num_of_iter_in_ot", 100)),
            reg=float(train_cfg.get("reg", 10.0)),
        )
        mae_loss = torch.nn.L1Loss().to(device)
        tv_loss = torch.nn.L1Loss(reduction="none").to(device)

        for epoch in range(1, int(args.epochs) + 1):
            loss_rows = []
            for batch in dataloaders["train"]:
                loss_rows.append(
                    run_train_batch(
                        model=model,
                        batch=batch,
                        optimizer=optimizer,
                        ot_loss=ot_loss,
                        tv_loss=tv_loss,
                        mae_loss=mae_loss,
                        wot=float(train_cfg.get("wot", 0.1)),
                        wtv=float(train_cfg.get("wtv", 0.01)),
                        device=device,
                    )
                )
            epoch_path = save_dir / f"{args.save_prefix}_{args.dataset_name}_epoch{epoch}.pth"
            torch.save(model.state_dict(), epoch_path)

    report = evaluate_dmcount_model(
        model,
        dataloaders["val"],
        device=device,
        dataset_name=args.dataset_name.upper(),
        result_type="adapted reproduction",
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
