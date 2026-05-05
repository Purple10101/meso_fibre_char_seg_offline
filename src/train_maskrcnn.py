"""
train_maskrcnn.py -- Mask R-CNN training for fibre instance segmentation
========================================================================

Usage:
    python train_maskrcnn.py --data_dir ./fibre_dataset --epochs 50

What this script does:
    1. Loads the fibre dataset via FibreDataset (manifest.json + RGB masks)
    2. Fine-tunes Mask R-CNN (ResNet-50-FPN-V2, COCO-pretrained)
    3. Validates every 2 epochs: COCO AP + M3-R01 coverage metric
    4. Saves best checkpoint by coverage fraction passing
    5. Produces training_log.csv (unified schema) and training_curves.png

Output:
    runs/fibre_maskrcnn/
        best.pth              -- best checkpoint by M3-R01 coverage
        epoch_NNN.pth         -- periodic checkpoints (every 5 epochs)
        training_log.csv      -- unified per-epoch log (same schema as YOLOv8 / SOLOv2)
        training_curves.png   -- loss + AP + coverage curves
        config.json           -- training hyperparameters

Requirements:
    pip install torch torchvision albumentations pycocotools tensorboard matplotlib
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FibreDataset, collate_fn, get_train_transforms, get_val_transforms
from model import build_model, count_parameters
from evaluate import evaluate_coco


# ================================================================================
#  Unified CSV schema (same as YOLOv8 and SOLOv2)
# ================================================================================

UNIFIED_FIELDS = [
    "epoch", "train_loss", "val_loss", "lr",
    "AP_mask", "AP50_mask", "AP75_mask", "AP_box",
    "coverage_fraction_passing", "coverage_mean",
    "coverage_median", "coverage_p5", "coverage_p95",
    "n_fibres",
]


# ================================================================================
#  Helpers
# ================================================================================

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [OK] checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    best_coverage = ckpt.get("best_coverage", 0.0)
    print(f"  [OK] resumed from epoch {ckpt['epoch']}  (best coverage={best_coverage:.4f})")
    return ckpt.get("epoch", 0), best_coverage


def log_epoch(csv_path, epoch, train_loss, lr, val_metrics=None, val_loss=None):
    cov_key = next(
        (k for k in (val_metrics or {}) if k.startswith("coverage_fraction_passing")),
        None,
    )
    row = {
        "epoch":      epoch,
        "train_loss": f"{train_loss:.6f}",
        "val_loss":   f"{val_loss:.6f}" if val_loss is not None else "",
        "lr":         f"{lr:.2e}",
        "AP_mask":    f"{val_metrics.get('AP_mask', '')}"    if val_metrics else "",
        "AP50_mask":  f"{val_metrics.get('AP50_mask', '')}"  if val_metrics else "",
        "AP75_mask":  f"{val_metrics.get('AP75_mask', '')}"  if val_metrics else "",
        "AP_box":     f"{val_metrics.get('AP_box', '')}"     if val_metrics else "",
        "coverage_fraction_passing":
            f"{val_metrics[cov_key]:.4f}" if val_metrics and cov_key else "",
        "coverage_mean":
            f"{val_metrics.get('coverage_mean', '')}"   if val_metrics else "",
        "coverage_median":
            f"{val_metrics.get('coverage_median', '')}" if val_metrics else "",
        "coverage_p5":
            f"{val_metrics.get('coverage_p5', '')}"     if val_metrics else "",
        "coverage_p95":
            f"{val_metrics.get('coverage_p95', '')}"    if val_metrics else "",
        "n_fibres":
            f"{val_metrics.get('n_fibres', '')}"        if val_metrics else "",
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ================================================================================
#  Plot generator
# ================================================================================

def plot_training_curves(csv_path, save_path):
    """Two-panel figure: (top) loss curves, (bottom) AP_mask + M3-R01 coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epochs, losses, val_losses = [], [], []
    ap_mask_vals, cov_vals = [], []
    val_epochs = []

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            epochs.append(ep)
            losses.append(float(row["train_loss"]))
            if row.get("val_loss", "").strip():
                val_losses.append((ep, float(row["val_loss"])))
            if row.get("AP_mask", "").strip():
                val_epochs.append(ep)
                ap_mask_vals.append(float(row["AP_mask"]))
                cov_vals.append(
                    float(row["coverage_fraction_passing"])
                    if row.get("coverage_fraction_passing", "").strip()
                    else float("nan")
                )

    fig, (ax_loss, ax_metrics) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Mask R-CNN -- Training Curves", fontsize=14)

    # Top panel: loss
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.plot(epochs, losses, color="#2563EB", linewidth=1.5, label="Train loss")
    if val_losses:
        ve, vl = zip(*val_losses)
        ax_loss.plot(ve, vl, color="#7C3AED", linewidth=1.5, linestyle="--",
                     label="Val loss")
    ax_loss.set_xlim(0, max(epochs))
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=10)

    # Bottom panel: AP_mask (left) + coverage fraction passing (right)
    ax_ap = ax_metrics
    ax_ap.set_xlabel("Epoch", fontsize=11)
    ax_ap.set_ylabel("AP$_{mask}$", color="#DC2626", fontsize=11)
    ax_ap.plot(val_epochs, ap_mask_vals, color="#DC2626", linewidth=1.5,
               marker="o", markersize=4, label="AP$_{mask}$")
    ax_ap.tick_params(axis="y", labelcolor="#DC2626")
    ax_ap.set_ylim(0, max(ap_mask_vals) * 1.15 if ap_mask_vals else 1.0)
    ax_ap.grid(True, alpha=0.3)

    ax_cov = ax_ap.twinx()
    ax_cov.set_ylabel("M3-R01 coverage fraction passing", color="#16A34A", fontsize=11)
    ax_cov.plot(val_epochs, cov_vals, color="#16A34A", linewidth=1.5,
                marker="s", markersize=4, linestyle="--", label="Coverage >=95%")
    ax_cov.axhline(0.70, color="#16A34A", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="Pass threshold (70%)")
    ax_cov.tick_params(axis="y", labelcolor="#16A34A")
    ax_cov.set_ylim(0, 1.05)

    lines_ap,  labels_ap  = ax_ap.get_legend_handles_labels()
    lines_cov, labels_cov = ax_cov.get_legend_handles_labels()
    ax_ap.legend(lines_ap + lines_cov, labels_ap + labels_cov,
                 loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] training curves saved -> {save_path}")


# ================================================================================
#  Training utilities
# ================================================================================

@torch.no_grad()
def compute_val_loss(model, loader, device, scaler):
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
        n += 1
    model.eval()
    return total / max(n, 1)


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, writer):
    model.train()
    total, t0 = 0.0, time.time()

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total += losses.item()
        global_step = epoch * len(loader) + step

        if step % 10 == 0:
            writer.add_scalar("train/loss_total",      losses.item(),                        global_step)
            writer.add_scalar("train/loss_classifier", loss_dict["loss_classifier"].item(),  global_step)
            writer.add_scalar("train/loss_box_reg",    loss_dict["loss_box_reg"].item(),     global_step)
            writer.add_scalar("train/loss_mask",       loss_dict["loss_mask"].item(),        global_step)
            writer.add_scalar("train/loss_objectness", loss_dict["loss_objectness"].item(),  global_step)
            writer.add_scalar("train/loss_rpn_box",    loss_dict["loss_rpn_box_reg"].item(), global_step)

        if step % 50 == 0:
            print(f"  step {step:4d}/{len(loader)}  "
                  f"loss={losses.item():.4f}  ({time.time() - t0:.0f}s elapsed)")

    return total / len(loader)


# ================================================================================
#  Main training function
# ================================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    writer  = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
    csv_path = os.path.join(args.out_dir, "training_log.csv")

    # -- Step 1: Datasets & loaders --
    train_ds = FibreDataset(args.data_dir, split="train",
                            transforms=get_train_transforms(args.image_size))
    val_ds   = FibreDataset(args.data_dir, split="val",
                            transforms=get_val_transforms(args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    # -- Step 2: Model --
    model = build_model(args.backbone, pretrained=True,
                        trainable_backbone_layers=args.trainable_layers)
    model.to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"\nStarting Mask R-CNN training for {args.epochs} epochs...")
    print(f"  Device      : {device}")
    print(f"  Image size  : {args.image_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Backbone    : {args.backbone}")
    print(f"  LR          : {args.lr:.2e}")
    print(f"  Output      : {args.out_dir}")
    print(f"  Train       : {len(train_ds)} samples")
    print(f"  Val         : {len(val_ds)} samples")
    print(f"  Parameters  : {total_params:,} total  {trainable_params:,} trainable")

    # -- Step 3: Optimiser & scheduler --
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if (not args.no_amp and device.type == "cuda") else None

    # -- Resume --
    start_epoch, best_coverage = 0, 0.0
    if args.resume:
        start_epoch, best_coverage = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -- Step 4: Training loop --
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'-' * 60}")
        print(f"Epoch {epoch}/{args.epochs - 1}   LR={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device, scaler, epoch, writer
        )
        scheduler.step()

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Validation every 2 epochs
        val_metrics = None
        val_loss    = None
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print("  Computing validation loss...")
            val_loss = compute_val_loss(model, val_loader, device, scaler)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)

            print("  Running COCO evaluation...")
            val_metrics = evaluate_coco(model, val_loader, device)
            ap_mask = val_metrics.get("AP_mask", 0.0)

            writer.add_scalar("val/AP_mask",   val_metrics.get("AP_mask",   0), epoch)
            writer.add_scalar("val/AP50_mask", val_metrics.get("AP50_mask", 0), epoch)
            writer.add_scalar("val/AP75_mask", val_metrics.get("AP75_mask", 0), epoch)
            writer.add_scalar("val/AP_box",    val_metrics.get("AP_box",    0), epoch)

            cov_key     = next((k for k in val_metrics if k.startswith("coverage_fraction_passing")), None)
            cov_passing = val_metrics.get(cov_key, 0.0) if cov_key else 0.0
            cov_mean    = val_metrics.get("coverage_mean",   0.0)
            cov_median  = val_metrics.get("coverage_median", 0.0)
            cov_p5      = val_metrics.get("coverage_p5",     0.0)
            cov_p95     = val_metrics.get("coverage_p95",    0.0)
            n_fibres    = val_metrics.get("n_fibres", 0)

            writer.add_scalar("val/coverage_fraction_passing", cov_passing, epoch)
            writer.add_scalar("val/coverage_mean",             cov_mean,    epoch)
            writer.add_scalar("val/coverage_median",           cov_median,  epoch)
            writer.add_scalar("val/coverage_p5",               cov_p5,      epoch)
            writer.add_scalar("val/coverage_p95",              cov_p95,     epoch)

            print(f"  val_loss={val_loss:.4f}  AP_mask={ap_mask:.4f}  "
                  f"AP50={val_metrics.get('AP50_mask', 0):.4f}  "
                  f"AP75={val_metrics.get('AP75_mask', 0):.4f}  "
                  f"AP_box={val_metrics.get('AP_box', 0):.4f}")
            print(f"  M3-R01 coverage: {cov_passing * 100:.1f}% of {n_fibres} fibres "
                  f"meet >=95% coverage  "
                  f"(mean={cov_mean:.3f}, median={cov_median:.3f}, "
                  f"p5={cov_p5:.3f}, p95={cov_p95:.3f})")

            if cov_passing >= 0.70:
                print(f"  [OK] M3-R01 PASS  ({cov_passing * 100:.1f}% >= 70%)")
            else:
                print(f"  [FAIL] M3-R01 FAIL  ({cov_passing * 100:.1f}% < 70%)")

            # Save best checkpoint by coverage
            if cov_passing > best_coverage:
                best_coverage = cov_passing
                save_checkpoint({
                    "epoch":         epoch,
                    "model":         model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "scheduler":     scheduler.state_dict(),
                    "best_coverage": best_coverage,
                    "metrics":       val_metrics,
                }, os.path.join(args.out_dir, "best.pth"))

        log_epoch(csv_path, epoch, avg_loss, scheduler.get_last_lr()[0],
                  val_metrics, val_loss=val_loss)

        if epoch % 5 == 0:
            save_checkpoint({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_coverage": best_coverage,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth"))

    # -- Step 5: Generate training curves --
    plot_path = os.path.join(args.out_dir, "training_curves.png")
    plot_training_curves(csv_path, plot_path)

    writer.close()

    print(f"\n[OK]  Mask R-CNN training complete.")
    print(f"   Best checkpoint : {os.path.join(args.out_dir, 'best.pth')}")
    print(f"   Training log    : {csv_path}")
    print(f"   Training curves : {plot_path}")
    print(f"   Best coverage   : {best_coverage:.4f}")


# ================================================================================
#  CLI
# ================================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Mask R-CNN on the fibre dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",          default=str(PROJECT_ROOT / "fibre_dataset"))
    p.add_argument("--out_dir",           default=str(PROJECT_ROOT / "runs" / "fibre_maskrcnn"))
    p.add_argument("--image_size",        type=int,   default=512)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--warmup_epochs",     type=int,   default=3)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--backbone",          default="maskrcnn_resnet50_fpn_v2",
                   choices=["maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2"])
    p.add_argument("--trainable_layers",  type=int,   default=3)
    p.add_argument("--resume",            default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_amp",            action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)