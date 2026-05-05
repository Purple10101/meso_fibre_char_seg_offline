"""
visualise_predictions.py -- Qualitative prediction vs ground-truth comparison
=============================================================================
Samples images from the val split and renders a multi-panel figure:

    [ Original | Ground Truth | Mask R-CNN | YOLOv8-seg ]

Each instance mask is drawn as a semi-transparent coloured fill + contour.
A per-panel annotation shows the instance count.

Usage:
    python visualise_predictions.py                          # 6 random val images
    python visualise_predictions.py --n_images 4
    python visualise_predictions.py --image_indices 0 5 12
    python visualise_predictions.py --models maskrcnn        # single model
    python visualise_predictions.py --score_thresh 0.4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import build_model


# --- Colour palette (12 visually distinct colours, RGB 0-255) -----------------

PALETTE = [
    (220,  50,  50), ( 50, 130, 220), ( 50, 180,  70), (220, 150,  50),
    (160,  50, 200), ( 50, 190, 180), (210, 200,  50), (200,  80, 160),
    (100, 160,  50), ( 80, 200, 200), (180,  80,  80), ( 80, 100, 200),
]


def _colour(idx):
    return PALETTE[idx % len(PALETTE)]


# --- Image helpers ------------------------------------------------------------

def load_image_and_gt(sample, data_dir, image_size):
    """Return (image_np [HxWx3 uint8], gt_masks [list of HxW uint8])."""
    img_path  = data_dir / Path(sample["image"])
    mask_path = data_dir / Path(sample["mask"])

    image    = np.array(Image.open(img_path).convert("RGB")
                        .resize((image_size, image_size), Image.BILINEAR))
    mask_rgb = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

    gt_masks = []
    for fibre in sample["fibres"]:
        rgb  = np.array(fibre["mask_rgb"], dtype=np.uint8)
        inst = np.all(mask_rgb == rgb, axis=-1).astype(np.uint8)
        if inst.sum() < 16:
            continue
        inst = cv2.resize(inst, (image_size, image_size),
                          interpolation=cv2.INTER_NEAREST)
        gt_masks.append(inst)

    return image, gt_masks


def overlay_masks(image_rgb, masks, gt_masks=None):
    """
    Renders coloured instance masks on a black background.

    If gt_masks is provided (prediction panels only), white GT contours are
    drawn on top so you can directly see where predictions match GT boundaries.
    Mismatches -- fills without a white outline (FP) or white outlines with
    no fill underneath (FN) -- stand out immediately.
    """
    out = np.zeros_like(image_rgb, dtype=np.uint8)

    for i, mask in enumerate(masks):
        out[mask.astype(bool)] = _colour(i)

    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, _colour(i), 1)

    if gt_masks is not None:
        for mask in gt_masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

    return out


# --- Per-model inference ------------------------------------------------------

def predict_maskrcnn(model, image_np, device, score_thresh):
    """Run Mask R-CNN on a uint8 HxWx3 image. Returns list of HxW binary masks."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = (t[c] - m) / s

    with torch.no_grad():
        output = model([t.to(device)])[0]

    binary_masks = []
    for mask, score in zip(output["masks"].cpu().numpy(),
                           output["scores"].cpu().numpy()):
        if score >= score_thresh:
            binary_masks.append((mask[0] > 0.5).astype(np.uint8))

    return binary_masks


def predict_yolov8(model, image_np, image_size, score_thresh):
    """Run YOLOv8-seg on a uint8 HxWx3 image. Returns list of HxW binary masks."""
    H, W    = image_np.shape[:2]
    results = model(image_np, imgsz=image_size, verbose=False)

    binary_masks = []
    if results and results[0].masks is not None:
        for mask, score in zip(results[0].masks.data.cpu().numpy(),
                               results[0].boxes.conf.cpu().numpy()):
            if score < score_thresh:
                continue
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            binary_masks.append((mask > 0.5).astype(np.uint8))

    return binary_masks


# --- Figure helpers -----------------------------------------------------------

def _show(ax, image, n_instances, col_label=None):
    ax.imshow(image)
    ax.axis("off")
    label = f"{n_instances} instance{'s' if n_instances != 1 else ''}"
    ax.text(0.02, 0.02, label, transform=ax.transAxes,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))
    if col_label:
        ax.set_title(col_label, fontsize=11, fontweight="bold", pad=5)


# --- Main ---------------------------------------------------------------------

def visualise(args):
    data_dir = Path(args.data_dir)

    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    val_samples = [s for s in manifest["samples"] if s["split"] == "val"]

    # -- Select images --
    if args.image_indices:
        indices  = list(args.image_indices)
    else:
        rng     = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(val_samples)),
                                    min(args.n_images, len(val_samples))))
    selected = [val_samples[i] for i in indices]

    # -- Load models --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maskrcnn_model = None
    if "maskrcnn" in args.models:
        maskrcnn_model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=False)
        ckpt_path = Path(args.maskrcnn_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            maskrcnn_model.load_state_dict(ckpt["model"])
            print(f"  Loaded Mask R-CNN  : {ckpt_path}")
        else:
            print(f"  WARNING: Mask R-CNN checkpoint not found -- {ckpt_path}")
        maskrcnn_model.to(device).eval()

    yolov8_model = None
    if "yolov8" in args.models:
        try:
            from ultralytics import YOLO
            yolo_path = Path(args.yolov8_ckpt)
            if yolo_path.exists():
                yolov8_model = YOLO(str(yolo_path))
                print(f"  Loaded YOLOv8-seg  : {yolo_path}")
            else:
                print(f"  WARNING: YOLOv8 checkpoint not found -- {yolo_path}")
        except ImportError:
            print("  WARNING: ultralytics not installed -- skipping YOLOv8")

    # -- Build column list --
    columns = ["Original", "Ground Truth"]
    if "maskrcnn" in args.models:
        columns.append("Mask R-CNN")
    if "yolov8" in args.models:
        columns.append("YOLOv8-seg")

    n_rows = len(selected)
    n_cols = len(columns)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)

    # -- Populate grid --
    for row_i, (sample, val_idx) in enumerate(zip(selected, indices)):
        image_np, gt_masks = load_image_and_gt(sample, data_dir, args.image_size)

        col = 0

        # Original
        _show(axes[row_i, col], image_np, n_instances=len(gt_masks),
              col_label="Original" if row_i == 0 else None)
        axes[row_i, col].set_ylabel(f"val[{val_idx}]", fontsize=9,
                                    rotation=0, labelpad=40, va="center")
        col += 1

        # Ground truth
        _show(axes[row_i, col], overlay_masks(image_np, gt_masks),
              n_instances=len(gt_masks),
              col_label="Ground Truth" if row_i == 0 else None)
        col += 1

        # Mask R-CNN
        if "maskrcnn" in args.models:
            if maskrcnn_model is not None:
                pred = predict_maskrcnn(maskrcnn_model, image_np, device, args.score_thresh)
                vis  = overlay_masks(image_np, pred, gt_masks=gt_masks)
            else:
                pred = []
                vis  = np.zeros_like(image_np)
            _show(axes[row_i, col], vis, n_instances=len(pred),
                  col_label="Mask R-CNN\n(white = GT outline)" if row_i == 0 else None)
            col += 1

        # YOLOv8-seg
        if "yolov8" in args.models:
            if yolov8_model is not None:
                pred = predict_yolov8(yolov8_model, image_np, args.image_size, args.score_thresh)
                vis  = overlay_masks(image_np, pred, gt_masks=gt_masks)
            else:
                pred = []
                vis  = np.zeros_like(image_np)
            _show(axes[row_i, col], vis, n_instances=len(pred),
                  col_label="YOLOv8-seg\n(white = GT outline)" if row_i == 0 else None)

    plt.suptitle(
        f"Fibre Segmentation -- Prediction vs Ground Truth  "
        f"(score thresh={args.score_thresh})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out_dir) / "prediction_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved -> {out_path}")


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise model predictions vs ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",       default=str(PROJECT_ROOT / "fibre_dataset"))
    p.add_argument("--out_dir",        default=str(PROJECT_ROOT / "benchmark_results"))
    p.add_argument("--maskrcnn_ckpt",  default=str(PROJECT_ROOT / "runs" / "fibre_maskrcnn" / "best.pth"))
    p.add_argument("--yolov8_ckpt",    default=str(PROJECT_ROOT / "runs" / "yolov8_seg" / "weights" / "best.pt"))
    p.add_argument("--models",  nargs="+", default=["maskrcnn", "yolov8"],
                   choices=["maskrcnn", "yolov8"])
    p.add_argument("--image_size",     type=int,   default=512)
    p.add_argument("--n_images",       type=int,   default=6,
                   help="Number of random val images to show")
    p.add_argument("--image_indices",  type=int,   nargs="+", default=None,
                   help="Specific val indices to show (overrides --n_images)")
    p.add_argument("--score_thresh",   type=float, default=0.5)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    visualise(parse_args())