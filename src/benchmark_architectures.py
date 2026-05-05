"""
benchmark_architectures.py -- Architecture Comparison for Fibre Segmentation
============================================================================

Benchmarks three instance segmentation architectures on the fibre dataset
and produces Table 21: Architecture Comparison on Fibre Segmentation.

Architectures:
  1. Mask R-CNN  (R50-FPN-V2)  -- torchvision
  2. YOLOv8-seg  (yolov8m-seg) -- ultralytics
  3. SOLOv2      (R50-FPN)     -- mmdetection

Usage:
    # Run all three (requires each framework installed)
    python benchmark_architectures.py --data_dir ./fibre_dataset --out_dir ./benchmark_results

    # Run only specific architectures
    python benchmark_architectures.py --models maskrcnn yolov8
    python benchmark_architectures.py --models solov2

    # Use existing checkpoints
    python benchmark_architectures.py \
        --maskrcnn_ckpt  runs/fibre_maskrcnn/best.pth \
        --yolov8_ckpt    runs/yolov8_seg/best.pt \
        --solov2_ckpt    runs/solov2/best.pth

Output:
    benchmark_results/
        table21_results.json    -- raw metrics for all models
        table21_results.csv     -- CSV version
        table21_latex.txt       -- LaTeX-formatted table ready for paper
        inference_timing.json   -- detailed per-image timing stats

Requirements:
    pip install torch torchvision pycocotools albumentations  # Mask R-CNN
    pip install ultralytics                                    # YOLOv8-seg
    pip install mmdet mmcv mmengine                             # SOLOv2

NOTE: You do NOT need all three frameworks installed at once.
      The script gracefully skips any architecture whose dependencies
      are missing and benchmarks the rest.
"""

import argparse
import csv
import json
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Local imports (your existing code) --------------------------------------
from dataset import FibreDataset, collate_fn, get_val_transforms
from model import build_model, count_parameters
from evaluate import evaluate_coco, compute_coverage_metric, compute_iou_metrics


# ================================================================================
#  1. MASK R-CNN  (torchvision)
# ================================================================================

def benchmark_maskrcnn(args):
    """Benchmark Mask R-CNN (R50-FPN-V2) using your existing code."""
    print("\n" + "=" * 70)
    print("  MASK R-CNN (ResNet-50 FPN V2)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Build model --
    model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=True)

    # -- Load checkpoint if available --
    if args.maskrcnn_ckpt and os.path.exists(args.maskrcnn_ckpt):
        ckpt = torch.load(args.maskrcnn_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint: {args.maskrcnn_ckpt}")
        epoch_info = f" (epoch {ckpt.get('epoch', '?')})"
    else:
        print("  WARNING: No checkpoint provided -- using COCO-pretrained weights.")
        print("  Results will NOT reflect fine-tuned performance on fibres.")
        epoch_info = " (COCO-pretrained)"

    model.to(device).eval()

    # -- Parameter count --
    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # -- Validation loader --
    val_ds = FibreDataset(
        args.data_dir, split="val",
        transforms=get_val_transforms(args.image_size),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  Val samples: {len(val_ds)}")

    # -- COCO evaluation --
    print("  Running COCO evaluation...")
    metrics = evaluate_coco(model, val_loader, device, score_thresh=0.3)

    # -- Inference timing --
    print("  Measuring inference time...")
    inference_ms = measure_inference_time(
        model, val_ds, device, args.image_size,
        n_warmup=10, n_measure=50, framework="torchvision",
    )

    coverage_pct = round(metrics.get("coverage_fraction_passing_at_0.95", 0.0) * 100, 1)

    result = {
        "architecture": "Mask R-CNN (R50-FPN-V2)",
        "AP_mask": round(metrics.get("AP_mask", 0.0), 4),
        "AP50_mask": round(metrics.get("AP50_mask", 0.0), 4),
        "coverage_pct": coverage_pct,

        "iou_mean": round(metrics.get("iou_mean", 0.0), 4),
        "iou_median": round(metrics.get("iou_median", 0.0), 4),
        "iou_p5": round(metrics.get("iou_p5", 0.0), 4),
        "iou_p95": round(metrics.get("iou_p95", 0.0), 4),
        "iou50_pct": round(metrics.get("iou50_fraction", 0.0) * 100, 1),
        "iou75_pct": round(metrics.get("iou75_fraction", 0.0) * 100, 1),

        "inference_ms": round(inference_ms, 1),
        "parameters": total_params,
        "params_M": round(total_params / 1e6, 1),
        "info": epoch_info.strip(),
    }
    print(
        f"\n  -> AP_mask={result['AP_mask']:.4f}"
        f"  AP50={result['AP50_mask']:.4f}"
        f"  Coverage>=95%={result['coverage_pct']}%"
        f"  IoU_mean={result['iou_mean']:.4f}"
        f"  IoU_median={result['iou_median']:.4f}"
        f"  IoU_p5={result['iou_p5']:.4f}"
        f"  IoU_p95={result['iou_p95']:.4f}"
        f"  IoU>=50%={result['iou50_pct']}%"
        f"  IoU>=75%={result['iou75_pct']}%"
        f"  Inference={result['inference_ms']:.1f}ms"
        f"  Params={result['params_M']}M"
    )
    return result


# ================================================================================
#  2. YOLOv8-SEG  (ultralytics)
# ================================================================================

def benchmark_yolov8(args):
    """Benchmark YOLOv8-seg using the ultralytics library."""
    print("\n" + "=" * 70)
    print("  YOLOv8-seg")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  SKIPPED: `ultralytics` not installed.")
        print("  Install with: pip install ultralytics")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Load model --
    print(f"  Checkpoint path : {args.yolov8_ckpt}")
    print(f"  Checkpoint found: {os.path.exists(args.yolov8_ckpt) if args.yolov8_ckpt else False}")
    if args.yolov8_ckpt and os.path.exists(args.yolov8_ckpt):
        model = YOLO(args.yolov8_ckpt)
        print(f"  Loaded checkpoint: {args.yolov8_ckpt}")
    else:
        # Use pretrained COCO model (for parameter count / inference speed)
        model = YOLO("yolov8m-seg.pt")
        print("  WARNING: No fine-tuned checkpoint -- using COCO-pretrained yolov8m-seg.")
        print("  AP metrics will NOT reflect fine-tuned performance on fibres.")

    # -- Parameter count --
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {total_params:,}")

    # -- Validation (ultralytics val) --
    print("  Running validation...")
    if args.yolov8_ckpt and os.path.exists(args.yolov8_ckpt):
        # Ultralytics needs a YOLO-format dataset yaml
        yaml_path = _ensure_yolo_yaml(args.data_dir)
        val_results = model.val(data=yaml_path, imgsz=args.image_size, device=device)
        ap_mask   = float(val_results.seg.map)      # mAP@0.5:0.95 masks
        ap50_mask = float(val_results.seg.map50)    # mAP@0.50 masks

        print("  Computing coverage metric...")
        cov = evaluate_yolov8_coverage(model, args.data_dir, args.image_size)

        coverage_pct = round(cov["fraction_passing"] * 100, 1)
        iou_mean = round(cov.get("mean_iou", 0.0), 4)
        iou_median = round(cov.get("median_iou", 0.0), 4)
        iou_p5 = round(cov.get("p5_iou", 0.0), 4)
        iou_p95 = round(cov.get("p95_iou", 0.0), 4)
        iou50_pct = round(cov.get("fraction_iou_passing_at_0.50", 0.0) * 100, 1)
        iou75_pct = round(cov.get("fraction_iou_passing_at_0.75", 0.0) * 100, 1)
    else:
        ap_mask = ap50_mask = 0.0
        coverage_pct = 0.0
        iou_mean = 0.0
        iou_median = 0.0
        iou_p5 = 0.0
        iou_p95 = 0.0
        iou50_pct = 0.0
        iou75_pct = 0.0
        print("  Skipping AP/coverage/IoU evaluation (no fine-tuned checkpoint).")

    # -- Inference timing --
    print("  Measuring inference time...")
    inference_ms = measure_inference_time_yolo(
        model, args.data_dir, args.image_size,
        n_warmup=10, n_measure=50,
    )

    result = {
        "architecture": "YOLOv8-seg",
        "AP_mask":       round(ap_mask, 4),
        "AP50_mask":     round(ap50_mask, 4),
        "coverage_pct":  coverage_pct,
        "inference_ms":  round(inference_ms, 1),
        "parameters":    total_params,
        "params_M":      round(total_params / 1e6, 1),
        "iou_mean": iou_mean,
        "iou_median": iou_median,
        "iou_p5": iou_p5,
        "iou_p95": iou_p95,
        "iou50_pct": iou50_pct,
        "iou75_pct": iou75_pct,
    }
    print(f"\n  -> AP_mask={result['AP_mask']:.4f}  AP50={result['AP50_mask']:.4f}"
          f"  Coverage>=95%={result['coverage_pct']}%"
          f"  Inference={result['inference_ms']:.1f}ms  Params={result['params_M']}M")
    return result


def _ensure_yolo_yaml(data_dir):
    """Create a YOLO-format dataset.yaml if it doesn't exist."""
    yaml_path = os.path.join(data_dir, "yolo_format/dataset.yaml")
    if os.path.exists(yaml_path):
        return yaml_path

    content = f"""# Auto-generated for benchmark
path: {os.path.abspath(data_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: fibre
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  Created dataset YAML: {yaml_path}")
    return yaml_path


def measure_inference_time_yolo(model, data_dir, image_size, n_warmup=10, n_measure=50):
    """Measure per-image inference time for YOLO models."""
    import glob
    from PIL import Image

    # Find validation images
    patterns = [
        os.path.join(data_dir, "images", "val", "*.png"),
        os.path.join(data_dir, "val", "images", "*.png"),
        os.path.join(data_dir, "images", "val", "*.jpg"),
    ]
    image_paths = []
    for pat in patterns:
        image_paths = glob.glob(pat)
        if image_paths:
            break

    if not image_paths:
        # Fallback: use manifest
        manifest_path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            image_paths = [
                os.path.join(data_dir, s["image"])
                for s in manifest["samples"] if s["split"] == "val"
            ]

    if not image_paths:
        print("  WARNING: No images found for timing.")
        return 0.0

    # Warmup
    for i in range(min(n_warmup, len(image_paths))):
        model.predict(image_paths[i % len(image_paths)], imgsz=image_size, verbose=False)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for i in range(n_measure):
        path = image_paths[i % len(image_paths)]
        t0 = time.perf_counter()
        model.predict(path, imgsz=image_size, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


def evaluate_yolov8_coverage(model, data_dir, image_size, coverage_threshold=0.95):
    """Compute coverage metric for YOLOv8-seg by running predict() on val images."""
    from PIL import Image as PILImage
    from pycocotools import mask as coco_mask_utils

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    val_samples = [s for s in manifest["samples"] if s["split"] == "val"]

    gt_annotations  = []
    pred_annotations = []
    ann_id = 1

    for img_idx, sample in enumerate(val_samples):
        img_path  = os.path.join(data_dir, sample["image"])
        mask_path = os.path.join(data_dir, sample["mask"])

        mask_rgb = np.array(PILImage.open(mask_path).convert("RGB"), dtype=np.uint8)
        H, W = mask_rgb.shape[:2]

        for fibre in sample["fibres"]:
            rgb = tuple(fibre["mask_rgb"])
            inst_mask = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            if inst_mask.sum() < 16:
                continue
            rle  = coco_mask_utils.encode(np.asfortranarray(inst_mask.astype(np.uint8)))
            area = float(coco_mask_utils.area(rle))
            rle["counts"] = rle["counts"].decode("utf-8")
            gt_annotations.append({
                "id": ann_id, "image_id": img_idx, "category_id": 1,
                "segmentation": rle, "area": area, "iscrowd": 0,
            })
            ann_id += 1

        results = model.predict(img_path, imgsz=image_size, verbose=False)
        if results and results[0].masks is not None:
            masks  = results[0].masks.data.cpu().numpy()   # NxHxW
            scores = results[0].boxes.conf.cpu().numpy()
            for j in range(len(scores)):
                if scores[j] < 0.3:
                    continue
                binary_mask = masks[j].astype(np.uint8)
                if binary_mask.shape != (H, W):
                    binary_mask = np.array(
                        PILImage.fromarray(binary_mask * 255).resize((W, H), PILImage.NEAREST)
                    ) // 255
                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                pred_annotations.append({
                    "image_id": img_idx, "category_id": 1,
                    "segmentation": rle, "score": float(scores[j]),
                })

    coverage_metrics = compute_coverage_metric(
        gt_annotations,
        pred_annotations,
        coverage_threshold,
    )

    iou_metrics = compute_iou_metrics(
        gt_annotations,
        pred_annotations,
    )

    return {
        **coverage_metrics,
        **iou_metrics,
    }


# ================================================================================
#  3. SOLOv2  (mmdetection)
# ================================================================================

def benchmark_solov2(args):
    """Benchmark SOLOv2 using mmdetection."""
    print("\n" + "=" * 70)
    print("  SOLOv2 (R50-FPN)")
    print("=" * 70)

    try:
        from mmdet.apis import init_detector, inference_detector
        from mmdet.evaluation import INSTANCE_OFFSET
        from mmengine.config import Config
        import mmdet
    except ImportError:
        print("  SKIPPED: `mmdet` / `mmcv` / `mmengine` not installed.")
        print("  Install with: pip install mmdet mmcv mmengine")
        return None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # -- Config --
    if args.solov2_config and os.path.exists(args.solov2_config):
        cfg_path = args.solov2_config
    else:
        # Try to find default mmdet config
        mmdet_dir = os.path.dirname(mmdet.__file__)
        cfg_path = os.path.join(
            mmdet_dir, ".mim", "configs", "solov2",
            "solov2_r50_fpn_1x_coco.py"
        )
        if not os.path.exists(cfg_path):
            print(f"  WARNING: SOLOv2 config not found at {cfg_path}")
            print("  Provide --solov2_config explicitly.")
            return None

    # -- Load model --
    if args.solov2_ckpt and os.path.exists(args.solov2_ckpt):
        model = init_detector(cfg_path, args.solov2_ckpt, device=device)
        print(f"  Loaded checkpoint: {args.solov2_ckpt}")
    else:
        model = init_detector(cfg_path, device=device)
        print("  WARNING: No fine-tuned checkpoint -- using default weights.")
        print("  AP metrics will NOT reflect fine-tuned performance on fibres.")

    # -- Parameter count --
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # -- Validation --
    print("  Running validation...")
    ap_mask, ap50_mask, coverage_pct = evaluate_solov2(model, args.data_dir, args.image_size, device)

    # -- Inference timing --
    print("  Measuring inference time...")
    inference_ms = measure_inference_time_mmdet(
        model, args.data_dir, args.image_size,
        n_warmup=10, n_measure=50,
    )

    result = {
        "architecture": "SOLOv2 (R50-FPN)",
        "AP_mask":       round(ap_mask, 4),
        "AP50_mask":     round(ap50_mask, 4),
        "coverage_pct":  coverage_pct,
        "inference_ms":  round(inference_ms, 1),
        "parameters":    total_params,
        "params_M":      round(total_params / 1e6, 1),
    }
    print(f"\n  -> AP_mask={result['AP_mask']:.4f}  AP50={result['AP50_mask']:.4f}"
          f"  Coverage>=95%={result['coverage_pct']}%"
          f"  Inference={result['inference_ms']:.1f}ms  Params={result['params_M']}M")
    return result


def evaluate_solov2(model, data_dir, image_size, device):
    """
    Run COCO-style evaluation on SOLOv2 predictions.
    Converts SOLOv2 outputs to COCO format and uses pycocotools.
    """
    from mmdet.apis import inference_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as coco_mask_utils

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_samples = [s for s in manifest["samples"] if s["split"] == "val"]

    gt_annotations = []
    pred_annotations = []
    images_info = []
    ann_id = 1

    for img_idx, sample in enumerate(val_samples):
        img_path = os.path.join(data_dir, sample["image"])
        mask_path = os.path.join(data_dir, sample["mask"])

        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGB")
        W, H = img.size
        images_info.append({"id": img_idx, "height": H, "width": W})

        # Ground truth
        mask_rgb = np.array(PILImage.open(mask_path).convert("RGB"), dtype=np.uint8)
        for fibre in sample["fibres"]:
            rgb = tuple(fibre["mask_rgb"])
            inst_mask = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            if inst_mask.sum() < 16:
                continue
            rle = coco_mask_utils.encode(np.asfortranarray(inst_mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            gt_annotations.append({
                "id": ann_id, "image_id": img_idx, "category_id": 1,
                "segmentation": rle, "bbox": coco_mask_utils.toBbox(rle).tolist(),
                "area": float(coco_mask_utils.area(rle)), "iscrowd": 0,
            })
            ann_id += 1

        # Prediction
        result = inference_detector(model, img_path)
        pred_instances = result.pred_instances

        if hasattr(pred_instances, "masks") and len(pred_instances.masks) > 0:
            masks = pred_instances.masks.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()

            for j in range(len(scores)):
                if scores[j] < 0.3:
                    continue
                binary_mask = masks[j].astype(np.uint8)
                if binary_mask.shape != (H, W):
                    from PIL import Image as PILImg2
                    binary_mask = np.array(
                        PILImg2.fromarray(binary_mask * 255).resize((W, H), PILImg2.NEAREST)
                    ) // 255
                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox = coco_mask_utils.toBbox(rle).tolist()
                pred_annotations.append({
                    "image_id": img_idx, "category_id": 1,
                    "segmentation": rle, "bbox": bbox,
                    "score": float(scores[j]),
                })

    if not gt_annotations:
        return 0.0, 0.0, 0.0

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images_info, "annotations": gt_annotations,
        "categories": [{"id": 1, "name": "fibre"}],
    }
    coco_gt.createIndex()

    if not pred_annotations:
        return 0.0, 0.0, 0.0

    coco_dt = coco_gt.loadRes(pred_annotations)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    cov = compute_coverage_metric(gt_annotations, pred_annotations)
    coverage_pct = round(cov["fraction_passing"] * 100, 1)

    return float(coco_eval.stats[0]), float(coco_eval.stats[1]), coverage_pct


def measure_inference_time_mmdet(model, data_dir, image_size, n_warmup=10, n_measure=50):
    """Measure per-image inference time for mmdetection models."""
    from mmdet.apis import inference_detector

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    val_samples = [s for s in manifest["samples"] if s["split"] == "val"]
    image_paths = [os.path.join(data_dir, s["image"]) for s in val_samples]

    if not image_paths:
        return 0.0

    # Warmup
    for i in range(min(n_warmup, len(image_paths))):
        inference_detector(model, image_paths[i % len(image_paths)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for i in range(n_measure):
        path = image_paths[i % len(image_paths)]
        t0 = time.perf_counter()
        inference_detector(model, path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


# ================================================================================
#  SHARED UTILITIES
# ================================================================================

def measure_inference_time(model, dataset, device, image_size,
                           n_warmup=10, n_measure=50, framework="torchvision"):
    """
    Measure per-image inference latency for torchvision models.
    Uses median of n_measure runs after n_warmup warm-up iterations.
    Includes torch.cuda.synchronize() for accurate GPU timing.
    """
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    model.eval()
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # Grab sample image paths
    n_total = min(n_warmup + n_measure, len(dataset))

    # Warmup
    with torch.no_grad():
        for i in range(min(n_warmup, len(dataset))):
            img, _ = dataset[i]
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)
            model([img.to(device)])
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for i in range(n_measure):
            idx = i % len(dataset)
            img, _ = dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)
            img_gpu = img.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            model([img_gpu])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)  # ms

    return float(np.median(times))


# ================================================================================
#  OUTPUT FORMATTERS
# ================================================================================

def format_results_table(results):
    """Print a formatted ASCII table matching Table 21."""
    print("\n")
    print("=" * 90)
    print("  Table 21: Architecture Comparison on Fibre Segmentation")
    print("=" * 90)
    header = (f"{'Architecture':<30} {'AP_mask':>10} {'AP50_mask':>10}"
              f" {'Cov>=95%':>9} {'IoU_med':>9} {'IoU75%':>8}"
              f" {'Infer (ms)':>12} {'Parameters':>12}")
    print(header)
    print("-" * 90)
    for r in results:
        params_str = f"{r['params_M']}M"
        cov_str    = f"{r.get('coverage_pct', 0.0):.1f}%"
        print(f"  {r['architecture']:<28} {r['AP_mask']:>10.4f} {r['AP50_mask']:>10.4f}"
              f" {cov_str:>9} {r.get('iou_median', 0.0):>9.4f}"
              f" {r.get('iou75_pct', 0.0):>7.1f}%"
              f" {r['inference_ms']:>12.1f} {params_str:>12}")
    print("=" * 90)


def save_latex_table(results, path):
    """Save LaTeX-formatted table to a .txt file."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Architecture Comparison on Fibre Segmentation}",
        r"\label{tab:arch_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{AP\_mask} & \textbf{AP50\_mask} & \textbf{Cov$\geq$95\%} & \textbf{Inference (ms)} & \textbf{Parameters} \\",
        r"\midrule",
    ]
    for r in results:
        cov_str = f"{r.get('coverage_pct', 0.0):.1f}\\%"
        lines.append(
            f"  {r['architecture']} & {r['AP_mask']:.4f} & {r['AP50_mask']:.4f} "
            f"& {cov_str} & {r['inference_ms']:.1f} & {r['params_M']}M \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table -> {path}")


def save_csv(results, path):
    """Save results as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "architecture",
            "AP_mask",
            "AP50_mask",
            "coverage_pct",
            "IoU_mean",
            "IoU_median",
            "IoU75_pct",
            "inference_ms",
            "parameters",
            "params_M",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, 0.0) for k in writer.fieldnames})
    print(f"  CSV table   -> {path}")


# ================================================================================
#  MAIN
# ================================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark instance segmentation architectures on fibre dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",    default=str(PROJECT_ROOT / "fibre_dataset"),
                   help="Path to fibre_dataset root (with manifest.json)")
    p.add_argument("--out_dir",     default=str(PROJECT_ROOT / "benchmark_results"),
                   help="Where to save results")
    p.add_argument("--image_size",  type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)

    # Which models to benchmark
    p.add_argument("--models", nargs="+",
                   default=["maskrcnn", "yolov8", "solov2"],
                   choices=["maskrcnn", "yolov8", "solov2"],
                   help="Which architectures to benchmark")

    # Checkpoint paths
    p.add_argument("--maskrcnn_ckpt", default=str(PROJECT_ROOT / "runs" / "fibre_maskrcnn" / "best.pth"),
                   help="Path to Mask R-CNN checkpoint")
    p.add_argument("--yolov8_ckpt",   default=str(PROJECT_ROOT / "runs" / "yolov8_seg" / "weights" / "best.pt"),
                   help="Path to YOLOv8-seg checkpoint (.pt)")
    p.add_argument("--solov2_ckpt",   default=None,
                   help="Path to SOLOv2 checkpoint (.pth)")
    p.add_argument("--solov2_config", default=None,
                   help="Path to SOLOv2 mmdet config (.py)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("==============================================================")
    print("|   Fibre Segmentation -- Architecture Benchmark              |")
    print("==============================================================")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Image size : {args.image_size}")
    print(f"  Models     : {', '.join(args.models)}")
    print(f"  Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    benchmark_fns = {
        "maskrcnn": benchmark_maskrcnn,
        "yolov8":   benchmark_yolov8,
        "solov2":   benchmark_solov2,
    }

    results = []
    for model_name in args.models:
        try:
            result = benchmark_fns[model_name](args)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"\n  ERROR benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n  No models were successfully benchmarked.")
        return

    # -- Display results --
    format_results_table(results)

    # -- Save outputs --
    json_path  = os.path.join(args.out_dir, "table21_results.json")
    csv_path   = os.path.join(args.out_dir, "table21_results.csv")
    latex_path = os.path.join(args.out_dir, "table21_latex.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON        -> {json_path}")

    save_csv(results, csv_path)
    save_latex_table(results, latex_path)

    print(f"\n[OK]  Benchmark complete -- results in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
