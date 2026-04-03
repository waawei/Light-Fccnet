"""Training script for Light-FCCNet reproduction."""
import argparse
import json
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from pack.models import build_model
from pack.data import GWHDDataset, URCDataset, MTCDataset
from pack.data.transforms import get_train_transforms, get_val_transforms
from pack.utils import (
    LightFCCLoss,
    descale_count,
    cal_mae,
    cal_mse,
    cal_mape,
    AverageMeter,
    filter_compatible_state_dict,
    combine_loss_terms,
    scale_density_target,
)
from pack.utils.losses import BaselineCountingLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Light-FCCNet training")
    parser.add_argument("--config", type=str, default="config/gwhd/config_gwhd_light_full.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def counting_collate_fn(batch):
    if len(batch) == 0:
        return {}

    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key == "points":
            collated[key] = values
        elif isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def create_dataloaders(config, train_transform, val_transform):
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    target_size = tuple(model_cfg.get("input_size", [256, 256]))
    sigma = data_cfg.get("sigma", 8)
    attention_radius = data_cfg.get("attention_radius", 2)
    urc_train_split_file = data_cfg.get("urc_train_split_file")
    urc_val_split_file = data_cfg.get("urc_val_split_file")
    mtc_train_split_file = data_cfg.get("mtc_train_split_file")
    mtc_val_split_file = data_cfg.get("mtc_val_split_file")
    train_sets, val_sets = [], []

    gwhd_train_csv = data_cfg.get("gwhd_train_csv")
    if gwhd_train_csv and os.path.exists(gwhd_train_csv):
        train_sets.append(
            GWHDDataset(
                gwhd_train_csv,
                data_cfg["gwhd_images_dir"],
                transform=train_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
        )
        val_sets.append(
            GWHDDataset(
                data_cfg["gwhd_val_csv"],
                data_cfg["gwhd_images_dir"],
                transform=val_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
        )

    urc_root = data_cfg.get("urc_root")
    if urc_root and os.path.exists(urc_root):
        if urc_train_split_file and urc_val_split_file and os.path.exists(urc_train_split_file) and os.path.exists(urc_val_split_file):
            train_sets.append(
                URCDataset(
                    urc_root,
                    split="train",
                    split_file=urc_train_split_file,
                    transform=train_transform,
                    target_size=target_size,
                    sigma=sigma,
                    attention_radius=attention_radius,
                )
            )
            val_sets.append(
                URCDataset(
                    urc_root,
                    split="val",
                    split_file=urc_val_split_file,
                    transform=val_transform,
                    target_size=target_size,
                    sigma=sigma,
                    attention_radius=attention_radius,
                )
            )
        else:
            train_sets.append(
                URCDataset(
                    urc_root,
                    split="train",
                    transform=train_transform,
                    target_size=target_size,
                    sigma=sigma,
                    attention_radius=attention_radius,
                )
            )
            val_sets.append(
                URCDataset(
                    urc_root,
                    split="test",
                    transform=val_transform,
                    target_size=target_size,
                    sigma=sigma,
                    attention_radius=attention_radius,
                )
            )

    mtc_root = data_cfg.get("mtc_root")
    if mtc_root and os.path.exists(mtc_root):
        if mtc_train_split_file and mtc_val_split_file and os.path.exists(mtc_train_split_file) and os.path.exists(mtc_val_split_file):
            mtc_train = MTCDataset(
                mtc_root,
                split="train",
                split_file=mtc_train_split_file,
                transform=train_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
            mtc_val = MTCDataset(
                mtc_root,
                split="val",
                split_file=mtc_val_split_file,
                transform=val_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
        elif os.path.exists(os.path.join(mtc_root, "train.txt")) and os.path.exists(os.path.join(mtc_root, "val.txt")):
            mtc_train = MTCDataset(
                mtc_root,
                split="train",
                transform=train_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
            mtc_val = MTCDataset(
                mtc_root,
                split="val",
                transform=val_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
        else:
            mtc_all = MTCDataset(
                mtc_root,
                split="train",
                transform=train_transform,
                target_size=target_size,
                sigma=sigma,
                attention_radius=attention_radius,
            )
            train_len = int(0.8 * len(mtc_all))
            val_len = len(mtc_all) - train_len
            mtc_train, mtc_val = torch.utils.data.random_split(mtc_all, [train_len, val_len])
        train_sets.append(mtc_train)
        val_sets.append(mtc_val)

    if len(train_sets) == 0:
        raise RuntimeError("No dataset was loaded. Please check GWHD/MTC/URC data paths in config.")

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        drop_last=True,
        collate_fn=counting_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        collate_fn=counting_collate_fn,
    )
    return train_loader, val_loader


def build_optimizer(config, model):
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {"type": "adam"})
    lr = train_cfg.get("learning_rate", 1e-4)

    if isinstance(opt_cfg, str):
        opt_type = opt_cfg.lower()
        opt_kwargs = {}
    else:
        opt_type = opt_cfg.get("type", "adam").lower()
        opt_kwargs = dict(opt_cfg)
        opt_kwargs.pop("type", None)

    if opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_kwargs.get("momentum", train_cfg.get("momentum", 0.9)),
            weight_decay=opt_kwargs.get("weight_decay", train_cfg.get("weight_decay", 1e-4)),
        )
    if opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(opt_kwargs.get("betas", (0.9, 0.999))),
            eps=opt_kwargs.get("eps", 1e-8),
            weight_decay=opt_kwargs.get("weight_decay", train_cfg.get("weight_decay", 1e-4)),
        )
    return optim.Adam(
        model.parameters(),
        lr=lr,
        betas=tuple(opt_kwargs.get("betas", (0.9, 0.999))),
        eps=opt_kwargs.get("eps", 1e-8),
        weight_decay=opt_kwargs.get("weight_decay", train_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(config, optimizer):
    sched_cfg = config.get("training", {}).get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "StepLR")).lower()
    if sched_type == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 100),
            eta_min=sched_cfg.get("eta_min", 1e-6),
        )
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=sched_cfg.get("step_size", 30),
        gamma=sched_cfg.get("gamma", 0.5),
    )


def build_criterion(config):
    train_cfg = config.get("training", {})
    loss_type = str(train_cfg.get("loss_type", "light_fcc")).lower()
    density_scale = float(train_cfg.get("density_scale", config.get("log_para", 1.0)))
    if loss_type != "light_fcc":
        raise ValueError(f"Unsupported training.loss_type: {loss_type}. Expected: light_fcc")
    use_p3_loss = bool(train_cfg.get("use_p3_loss", False))
    if use_p3_loss:
        ssim_c = train_cfg.get("ssim_c")
        if ssim_c is None:
            ssim_c = train_cfg.get("ssim_c1", 1e-4)
        criterion = LightFCCLoss(
            alpha=float(train_cfg.get("alpha", 0.1)),
            density_scale=density_scale,
            ssim_window=int(train_cfg.get("ssim_window", 11)),
            ssim_c=float(ssim_c),
        )
    else:
        criterion = BaselineCountingLoss(density_scale=density_scale)
    criterion.raw_density_loss_weight = float(train_cfg.get("raw_density_loss_weight", 0.0))
    criterion.attention_loss_weight = float(train_cfg.get("attention_loss_weight", 0.0))
    return criterion


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    use_amp = scaler is not None
    loss_meter = AverageMeter()
    aux_density_meter = AverageMeter()
    aux_attention_meter = AverageMeter()
    pbar = tqdm(loader, desc="Train")
    raw_density_loss_weight = float(getattr(criterion, "raw_density_loss_weight", 0.0))
    attention_loss_weight = float(getattr(criterion, "attention_loss_weight", 0.0))
    density_scale = float(getattr(criterion, "density_scale", 1.0))
    attention_branch_enabled = bool(getattr(model, "use_p2", True))

    for batch in pbar:
        image = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        gt_attention = batch.get("attention_mask")
        if gt_attention is not None:
            gt_attention = gt_attention.to(device)
        bs = image.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            final_density, attention, raw_density = model(image)
            loss, light_terms = criterion(
                final_density,
                gt_density,
                gt_count=batch.get("count"),
            )
            density_loss = light_terms["l2"]
            ssim_loss = light_terms.get("ssim")
            aux_density_loss = None
            aux_attention_loss = None
            if raw_density_loss_weight > 0.0:
                scaled_gt_density = scale_density_target(gt_density, density_scale)
                aux_density_loss = F.mse_loss(raw_density, scaled_gt_density)
                loss = combine_loss_terms(loss, aux_density_loss, raw_density_loss_weight)
            if attention_branch_enabled and attention_loss_weight > 0.0 and gt_attention is not None:
                aux_attention_loss = F.binary_cross_entropy(attention, gt_attention)
                loss = combine_loss_terms(loss, aux_attention_loss, attention_loss_weight)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_meter.update(loss.item(), bs)
        postfix = {"loss": f"{loss_meter.avg:.4f}"}
        if density_loss is not None:
            postfix["d"] = f"{float(density_loss.item()):.4f}"
        if ssim_loss is not None:
            postfix["s"] = f"{float(ssim_loss.item()):.4f}"
        if aux_density_loss is not None:
            aux_density_meter.update(float(aux_density_loss.item()), bs)
            postfix["rd"] = f"{aux_density_meter.avg:.4f}"
        if aux_attention_loss is not None:
            aux_attention_meter.update(float(aux_attention_loss.item()), bs)
            postfix["att"] = f"{aux_attention_meter.avg:.4f}"
        pbar.set_postfix(postfix)

    return {"loss": loss_meter.avg}


@torch.no_grad()
def validate(model, loader, device, density_scale: float = 1.0):
    model.eval()
    pred_counts = []
    gt_counts = []
    for batch in tqdm(loader, desc="Val"):
        image = batch["image"].to(device)
        gt_count = batch["count"].to(device)
        final_density, _, _ = model(image)
        pred_count = final_density.sum(dim=(1, 2, 3))
        pred_count = descale_count(pred_count, density_scale)
        pred_counts.extend(pred_count.cpu().numpy().tolist())
        gt_counts.extend(gt_count.cpu().numpy().tolist())

    pred_counts = np.array(pred_counts, dtype=np.float32)
    gt_counts = np.array(gt_counts, dtype=np.float32)
    mae = cal_mae(pred_counts, gt_counts)
    mse = cal_mse(pred_counts, gt_counts)
    mape = cal_mape(pred_counts, gt_counts)
    return {"MAE": mae, "MSE": mse, "MAPE": mape}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    paths = cfg.setdefault("paths", {})
    ckpt_dir = paths.get("checkpoint_dir", "./checkpoints/new_fccnet")
    log_dir = paths.get("log_dir", "./logs/new_fccnet")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = None
    try:
        writer = SummaryWriter(log_dir=log_dir)
    except Exception as e:
        print(f"WARNING: TensorBoard disabled due to writer init error: {e}")

    train_tf = get_train_transforms(cfg)
    val_tf = get_val_transforms()
    train_loader, val_loader = create_dataloaders(cfg, train_tf, val_tf)

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler() if cfg.get("use_amp", True) else None
    train_cfg = cfg.get("training", {})

    start_epoch = 1
    best_mae = float("inf")
    best_metrics = {"MAE": best_mae, "MSE": None, "MAPE": None}
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if bool(train_cfg.get("init_weights_only", False)):
            matched_state, skipped_state = filter_compatible_state_dict(model.state_dict(), checkpoint["model_state_dict"])
            model.load_state_dict(matched_state, strict=False)
            print(
                f"Initialized model from checkpoint with {len(matched_state)} compatible tensors; "
                f"skipped {len(skipped_state)} incompatible tensors."
            )
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_mae = checkpoint.get("best_mae", best_mae)
            best_metrics = checkpoint.get("best_metrics", best_metrics)

    eval_cfg = cfg.get("eval", {})
    num_epochs = train_cfg.get("num_epochs", 100)
    density_scale = float(train_cfg.get("density_scale", cfg.get("log_para", 1.0)))
    val_freq = eval_cfg.get("val_freq", 1)
    save_freq = eval_cfg.get("save_freq", 10)

    for epoch in range(start_epoch, num_epochs + 1):
        train_log = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_log['loss']:.6f} | LR: {current_lr:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Train/Loss", train_log["loss"], epoch)
            writer.add_scalar("Train/LR", current_lr, epoch)

        if epoch % val_freq == 0:
            val_log = validate(model, val_loader, device, density_scale=density_scale)
            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"Val MAE: {val_log['MAE']:.6f} | Val MSE: {val_log['MSE']:.6f} | Val MAPE: {val_log['MAPE']:.6f} | "
                f"Best MAE: {min(best_mae, val_log['MAE']):.6f}"
            )
            if writer is not None:
                writer.add_scalar("Val/MAE", val_log["MAE"], epoch)
                writer.add_scalar("Val/MSE", val_log["MSE"], epoch)
                writer.add_scalar("Val/MAPE", val_log["MAPE"], epoch)

            if val_log["MAE"] < best_mae:
                best_mae = val_log["MAE"]
                best_metrics = dict(val_log)
                print(
                    f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                    f"New best model saved with MAE: {best_mae:.6f}"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mae": best_mae,
                        "best_metrics": best_metrics,
                        "config": cfg,
                    },
                    os.path.join(ckpt_dir, "best_model.pth"),
                )
                with open(os.path.join(ckpt_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, "best_metrics": best_metrics}, f, ensure_ascii=False, indent=2)

        if epoch % save_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "last_val_metrics": val_log if epoch % val_freq == 0 else None,
                    "config": cfg,
                },
                os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    if writer is not None:
        writer.close()
    print(f"Training finished. Best MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main()
