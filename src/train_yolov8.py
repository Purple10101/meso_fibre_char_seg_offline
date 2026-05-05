"""
train_yolov8.py -- YOLOv8-seg training for fibre instance segmentation
======================================================================

Converts the fibre dataset (manifest.json + RGB instance masks) into
YOLO segmentation format, then trains a YOLOv8m-seg model.

Usage:
    python train_yolov8.py --data_dir ./fibre_dataset --epochs 100

What this script does:
    1. Reads manifest.json and RGB instance masks
    2. Converts to YOLO-seg format:  <class> <x1> <y1> <x2> <y2> ... (polygon)
    3. Creates a dataset.yaml for ultralytics
    4. Trains yolov8m-seg with COCO-pretrained weights
    5. Converts ultralytics results.csv -> training_log.csv (unified format)
    6. Generates training_curves.png

Output:
    runs/yolov8_seg/
        weights/best.pt       -- best checkpoint (use with benchmark script)
        weights/last.pt       -- final checkpoint
        training_log.csv      -- unified per-epoch log (same schema as Mask R-CNN)
        training_curves.png   -- loss + AP curves
        results.csv           -- ultralytics native log (kept for reference)

Requirements:
    pip install ultralytics opencv-python-headless numpy matplotlib
"""

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ================================================================================
#  STEP 1: Convert fibre dataset -> YOLO segmentation format
# ================================================================================

def mask_to_polygons(binary_mask, epsilon_frac=0.002):
    """
    Convert a binary mask to a list of polygon contours in normalised
    [0, 1] coordinates, suitable for YOLO segmentation format.

    Args:
        binary_mask : np.ndarray HxW uint8 (255 = foreground)
        epsilon_frac: contour approximation tolerance as fraction of perimeter

    Returns:
        list of np.ndarray, each shape (N, 2) with normalised (x, y) coords.
        Returns only polygons with >= 3 vertices.
    """
    H, W = binary_mask.shape[:2]
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 10:  # skip tiny artefacts
            continue
        approx = cv2.approxPolyDP(cnt, epsilon_frac * perimeter, True)
        if len(approx) < 3:
            continue
        # Normalise to [0, 1]
        pts = approx.squeeze().astype(np.float64)
        if pts.ndim == 1:
            continue
        pts[:, 0] /= W
        pts[:, 1] /= H
        polygons.append(pts)

    return polygons


def convert_dataset_to_yolo(data_dir, yolo_dir, min_area=16):
    """
    Read manifest.json, decode RGB instance masks into per-instance polygons,
    and write YOLO-seg format label files.
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Converting {len(manifest['samples'])} samples to YOLO-seg format...")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "instances": 0, "skipped": 0}

    for sample in manifest["samples"]:
        split = sample["split"]
        img_path  = os.path.join(data_dir, sample["image"])
        mask_path = os.path.join(data_dir, sample["mask"])

        stem = Path(sample["image"]).stem
        ext  = Path(sample["image"]).suffix

        dst_img = os.path.join(yolo_dir, "images", split, f"{stem}{ext}")
        if not os.path.exists(dst_img):
            shutil.copy2(img_path, dst_img)

        mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_rgb is None:
            print(f"  WARNING: Could not read mask {mask_path}")
            continue
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        label_lines = []
        for fibre in sample["fibres"]:
            rgb = tuple(fibre["mask_rgb"])
            inst_mask = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            area = inst_mask.sum()
            if area < min_area:
                stats["skipped"] += 1
                continue

            binary = (inst_mask.astype(np.uint8) * 255)
            polygons = mask_to_polygons(binary)

            for poly in polygons:
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                label_lines.append(f"0 {coords}")
                stats["instances"] += 1

        label_path = os.path.join(yolo_dir, "labels", split, f"{stem}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        stats[split] += 1

    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    abs_yolo = os.path.abspath(yolo_dir)
    with open(yaml_path, "w") as f:
        f.write(f"# Fibre instance segmentation dataset (YOLO-seg format)\n")
        f.write(f"# Auto-generated by train_yolov8.py\n\n")
        f.write(f"path: {abs_yolo}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"names:\n")
        f.write(f"  0: fibre\n")

    print(f"  Conversion complete:")
    print(f"    Train : {stats['train']} images")
    print(f"    Val   : {stats['val']} images")
    print(f"    Test  : {stats['test']} images")
    print(f"    Total instances: {stats['instances']}  (skipped {stats['skipped']} tiny masks)")
    print(f"    YAML  : {yaml_path}")

    return yaml_path


# ================================================================================
#  TRAINING LOG -- Unified CSV format matching Mask R-CNN's training_log.csv
# ================================================================================

UNIFIED_FIELDS = [
    "epoch", "train_loss", "val_loss", "lr",
    "AP_mask", "AP50_mask", "AP75_mask", "AP_box",
]


def convert_ultralytics_csv(results_csv_path, output_csv_path):
    """
    Convert ultralytics results.csv -> unified training_log.csv.

    Ultralytics columns (stripped):
        epoch, train/box_loss, train/seg_loss, train/cls_loss, train/dfl_loss,
        metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M),
        metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
        val/box_loss, val/seg_loss, val/cls_loss, val/dfl_loss, lr/pg0, lr/pg1, lr/pg2

    Unified mapping:
        train_loss = sum(train/box_loss, train/seg_loss, train/cls_loss, train/dfl_loss)
        val_loss   = sum(val/box_loss, val/seg_loss, val/cls_loss, val/dfl_loss)
        AP_mask    = metrics/mAP50-95(M)
        AP50_mask  = metrics/mAP50(M)
        AP_box     = metrics/mAP50-95(B)
        lr         = lr/pg0
    """
    if not os.path.exists(results_csv_path):
        print(f"  WARNING: ultralytics results.csv not found at {results_csv_path}")
        return

    rows_out = []

    with open(results_csv_path, "r") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [col.strip() for col in reader.fieldnames]

        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}

            try:
                epoch = int(row.get("epoch", 0))

                train_loss_keys = ["train/box_loss", "train/seg_loss",
                                   "train/cls_loss", "train/dfl_loss"]
                train_loss = sum(float(row.get(k, 0)) for k in train_loss_keys)

                val_loss_keys = ["val/box_loss", "val/seg_loss",
                                 "val/cls_loss", "val/dfl_loss"]
                val_loss = sum(float(row.get(k, 0)) for k in val_loss_keys)

                lr = float(row.get("lr/pg0", row.get("lr/pg1", 0)))

                ap_mask   = float(row.get("metrics/mAP50-95(M)", 0))
                ap50_mask = float(row.get("metrics/mAP50(M)", 0))
                ap_box    = float(row.get("metrics/mAP50-95(B)", 0))

                rows_out.append({
                    "epoch":      epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss":   f"{val_loss:.6f}",
                    "lr":         f"{lr:.2e}",
                    "AP_mask":    f"{ap_mask:.6f}" if ap_mask > 0 else "",
                    "AP50_mask":  f"{ap50_mask:.6f}" if ap50_mask > 0 else "",
                    "AP75_mask":  "",
                    "AP_box":     f"{ap_box:.6f}" if ap_box > 0 else "",
                })
            except (ValueError, KeyError) as e:
                print(f"  WARNING: could not parse row: {e}")
                continue

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  [OK] Unified training log ({len(rows_out)} epochs) -> {output_csv_path}")


# ================================================================================
#  PLOT GENERATOR -- matches Mask R-CNN plot style
# ================================================================================

def plot_training_curves(csv_path, save_path, title="YOLOv8-seg"):
    """Generate dual-axis figure: training+val loss (left) + validation AP_mask (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_losses, val_losses = [], [], []
    ap_mask_vals, val_epochs = [], []

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            epochs.append(ep)
            train_losses.append(float(row["train_loss"]))
            if row.get("val_loss"):
                val_losses.append(float(row["val_loss"]))
            else:
                val_losses.append(None)
            if row["AP_mask"]:
                val_epochs.append(ep)
                ap_mask_vals.append(float(row["AP_mask"]))

    if not epochs:
        print(f"  WARNING: No data to plot in {csv_path}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis -- Training Loss + Val Loss
    colour_train = "#2563EB"
    colour_val   = "#7C3AED"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color=colour_train, fontsize=12)
    ax1.plot(epochs, train_losses, color=colour_train, linewidth=1.5,
             label="Train Loss")
    valid_val = [(e, v) for e, v in zip(epochs, val_losses) if v is not None]
    if valid_val:
        ve, vl = zip(*valid_val)
        ax1.plot(ve, vl, color=colour_val, linewidth=1.5, linestyle="--",
                 label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=colour_train)
    ax1.set_xlim(0, max(epochs))
    ax1.grid(True, alpha=0.3)

    # Right axis -- Validation AP_mask
    ax2 = ax1.twinx()
    colour_ap = "#DC2626"
    ax2.set_ylabel("Validation AP$_{mask}$", color=colour_ap, fontsize=12)
    if ap_mask_vals:
        ax2.plot(val_epochs, ap_mask_vals, color=colour_ap, linewidth=1.5,
                 marker="o", markersize=4, label="Val AP$_{mask}$")
        ax2.set_ylim(0, max(ap_mask_vals) * 1.15)
    ax2.tick_params(axis="y", labelcolor=colour_ap)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    plt.title(f"{title} -- Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] training curves saved -> {save_path}")


# ================================================================================
#  M3-R01 COVERAGE EVALUATION -- post-training, on best checkpoint
# ================================================================================

def evaluate_coverage_yolo(yolo_model, data_dir, image_size,
                           score_thresh=0.3, coverage_threshold=0.95):
    """
    Run the M3-R01 coverage metric on the val split using a trained YOLO model.
    Converts YOLO predictions to RLE annotations and feeds them into the same
    compute_coverage_metric used by Mask R-CNN evaluation.
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    from torch.utils.data import DataLoader
    from dataset import FibreDataset, collate_fn, get_val_transforms
    from evaluate import compute_coverage_metric
    from pycocotools import mask as coco_mask_utils

    val_ds = FibreDataset(data_dir, split="val",
                          transforms=get_val_transforms(image_size))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    gt_anns, pred_anns = [], []
    ann_id = 1

    for images, targets in val_loader:
        for img_tensor, target in zip(images, targets):
            image_id = int(target["image_id"].item())
            H, W = img_tensor.shape[-2], img_tensor.shape[-1]

            # Ground truth -- encode each instance mask as RLE
            for mask in target["masks"].numpy():
                rle = coco_mask_utils.encode(
                    np.asfortranarray(mask.astype(np.uint8))
                )
                rle["counts"] = rle["counts"].decode("utf-8")
                area = float(coco_mask_utils.area(rle))
                if area == 0:
                    continue
                gt_anns.append({
                    "id": ann_id, "image_id": image_id,
                    "segmentation": rle, "area": area,
                })
                ann_id += 1

            # YOLO inference -- convert float CHW tensor [0,1] -> uint8 HWC
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255
                      ).clip(0, 255).astype(np.uint8)
            results = yolo_model(img_np, verbose=False, imgsz=image_size)

            for result in results:
                if result.masks is None:
                    continue
                masks_data = result.masks.data.cpu().numpy()  # NxH'xW'
                scores     = result.boxes.conf.cpu().numpy()

                for mask_arr, score in zip(masks_data, scores):
                    if score < score_thresh:
                        continue
                    if mask_arr.shape != (H, W):
                        mask_arr = cv2.resize(
                            mask_arr.astype(np.float32), (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    binary = (mask_arr > 0.5).astype(np.uint8)
                    rle = coco_mask_utils.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    pred_anns.append({
                        "image_id": image_id,
                        "segmentation": rle,
                        "score": float(score),
                    })

    return compute_coverage_metric(gt_anns, pred_anns, coverage_threshold)


def append_coverage_to_log(csv_path, coverage_metrics):
    """
    Add coverage columns to the unified CSV and write the values into the
    last row (representing the best-checkpoint evaluation after training).
    All other rows get empty strings for those columns.
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return

    cov_fields = [
        "coverage_fraction_passing", "coverage_mean", "coverage_median",
        "coverage_p5", "coverage_p95", "n_fibres",
    ]
    all_fields = existing_fields + [f for f in cov_fields if f not in existing_fields]

    for row in rows:
        for field in cov_fields:
            row.setdefault(field, "")

    last = rows[-1]
    last["coverage_fraction_passing"] = f"{coverage_metrics['fraction_passing']:.4f}"
    last["coverage_mean"]             = f"{coverage_metrics['mean_coverage']:.4f}"
    last["coverage_median"]           = f"{coverage_metrics['median_coverage']:.4f}"
    last["coverage_p5"]               = f"{coverage_metrics['p5_coverage']:.4f}"
    last["coverage_p95"]              = f"{coverage_metrics['p95_coverage']:.4f}"
    last["n_fibres"]                  = str(coverage_metrics["n_fibres"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [OK] Coverage results written to {csv_path} (last row)")


# ================================================================================
#  STEP 2: Train YOLOv8-seg
# ================================================================================

def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("Install with:  pip install ultralytics")
        sys.exit(1)

    # -- Convert dataset --
    yolo_dir  = os.path.join(args.data_dir, "yolo_format")
    yaml_path = convert_dataset_to_yolo(args.data_dir, yolo_dir, min_area=args.min_area)

    # -- Load pretrained model --
    model_variant = args.model
    print(f"\nLoading {model_variant} with COCO-pretrained weights...")
    model = YOLO(f"{model_variant}.pt")

    # -- Train --
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Image size  : {args.image_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Output      : {args.out_dir}")

    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        lr0=args.lr,
        lrf=0.01,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,

        # Augmentation
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        flipud=0.5, fliplr=0.5,
        degrees=90.0, scale=0.3,
        mosaic=1.0, mixup=0.1,

        # Training settings
        optimizer="AdamW",
        cos_lr=True,
        amp=True,
        workers=args.num_workers,
        patience=20,
        save_period=5,
        val=True,
        plots=True,

        # Output -- use resolved absolute path so CWD doesn't affect where files land
        project=str(Path(args.out_dir).resolve().parent),
        name=Path(args.out_dir).name,
        exist_ok=True,
    )

    # -- Resolve the actual save directory ultralytics used --
    # model.trainer.save_dir is the ground truth -- it may differ from args.out_dir
    # on Windows or when ultralytics auto-increments a clashing directory name.
    save_dir = str(model.trainer.save_dir)
    print(f"\n  Ultralytics save dir: {save_dir}")

    # -- Convert ultralytics results.csv -> unified training_log.csv --
    print("\nConverting training log to unified format...")
    results_csv = os.path.join(save_dir, "results.csv")
    unified_csv = os.path.join(save_dir, "training_log.csv")
    convert_ultralytics_csv(results_csv, unified_csv)

    # -- Generate training curves --
    plot_path = os.path.join(save_dir, "training_curves.png")
    if os.path.exists(unified_csv):
        plot_training_curves(unified_csv, plot_path, title="YOLOv8-seg")
    else:
        print(f"  WARNING: training_log.csv not found -- results.csv missing from {save_dir}")

    # -- Final validation --
    print("\nRunning final validation on best checkpoint...")
    best_path = os.path.join(save_dir, "weights", "best.pt")
    if os.path.exists(best_path):
        best_model = YOLO(best_path)
        val_results = best_model.val(data=yaml_path, imgsz=args.image_size)
        print(f"\n  Final Results:")
        print(f"    Mask  mAP@0.5:0.95 = {val_results.seg.map:.4f}")
        print(f"    Mask  mAP@0.5      = {val_results.seg.map50:.4f}")
        print(f"    Box   mAP@0.5:0.95 = {val_results.box.map:.4f}")

        # -- M3-R01 coverage evaluation --
        print("\nEvaluating M3-R01 coverage metric on val set...")
        coverage_metrics = evaluate_coverage_yolo(
            best_model, args.data_dir, args.image_size,
        )
        cov_passing = coverage_metrics["fraction_passing"]
        print(f"  M3-R01 coverage: {cov_passing * 100:.1f}% of "
              f"{coverage_metrics['n_fibres']} fibres meet >=95% coverage")
        print(f"  (mean={coverage_metrics['mean_coverage']:.3f}, "
              f"median={coverage_metrics['median_coverage']:.3f}, "
              f"p5={coverage_metrics['p5_coverage']:.3f}, "
              f"p95={coverage_metrics['p95_coverage']:.3f})")
        if cov_passing >= 0.70:
            print(f"  [OK] M3-R01 PASS  ({cov_passing * 100:.1f}% >= 70%)")
        else:
            print(f"  [FAIL] M3-R01 FAIL  ({cov_passing * 100:.1f}% < 70%)")

        append_coverage_to_log(unified_csv, coverage_metrics)
    else:
        print(f"  WARNING: best.pt not found at {best_path}")

    print(f"\n[OK]  YOLOv8-seg training complete.")
    print(f"   Best checkpoint : {best_path}")
    print(f"   Training log    : {unified_csv}")
    print(f"   Training curves : {plot_path}")


# ================================================================================
#  CLI
# ================================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLOv8-seg on the fibre dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",       default=str(PROJECT_ROOT / "fibre_dataset"))
    p.add_argument("--out_dir",        default=str(PROJECT_ROOT / "runs" / "yolov8_seg"))
    p.add_argument("--model",          default="yolov8m-seg",
                   choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg",
                            "yolov8l-seg", "yolov8x-seg"])
    p.add_argument("--image_size",     type=int, default=512)
    p.add_argument("--epochs",         type=int, default=50)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=5e-4)
    p.add_argument("--warmup_epochs",  type=int, default=5)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--min_area",       type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
