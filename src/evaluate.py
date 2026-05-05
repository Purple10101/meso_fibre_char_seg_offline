"""
evaluate.py -- COCO-style evaluation for fibre instance segmentation
====================================================================

Returns COCO-style AP metrics for both masks and boxes, plus custom
fibre-level coverage and matched-IoU metrics.

COCO metrics:
    AP_mask, AP50_mask, AP75_mask
    AP_box,  AP50_box,  AP75_box

Custom fibre-level metrics:
    coverage_fraction_passing_at_<threshold>
    coverage_mean
    coverage_median
    coverage_p5
    coverage_p95
    n_fibres

    iou_mean
    iou_median
    iou_p5
    iou_p95
    iou50_fraction
    iou75_fraction
    n_fibres_iou
"""

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils


# =============================================================================
# Prediction collector
# =============================================================================

@torch.no_grad()
def collect_predictions(model, loader, device, score_thresh=0.3):
    """
    Run inference over a DataLoader and collect predictions and ground truths
    in COCO annotation format for pycocotools evaluation.
    """
    model.eval()

    gt_annotations = []
    pred_annotations = []
    images_info = []
    ann_id = 1

    for batch_idx, (images, targets) in enumerate(loader):
        images_gpu = [img.to(device) for img in images]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images_gpu)

        for i, (output, target) in enumerate(zip(outputs, targets)):
            image_id = int(target["image_id"].item())
            H, W = images[i].shape[-2], images[i].shape[-1]

            images_info.append({
                "id": image_id,
                "height": H,
                "width": W,
            })

            # -----------------------------------------------------------------
            # Ground truth annotations
            # -----------------------------------------------------------------
            gt_masks = target["masks"].cpu().numpy()    # N x H x W
            gt_labels = target["labels"].cpu().numpy()  # N

            for j in range(len(gt_masks)):
                gt_mask = gt_masks[j].astype(np.uint8)

                if gt_mask.sum() == 0:
                    continue

                rle = coco_mask_utils.encode(np.asfortranarray(gt_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                bbox = coco_mask_utils.toBbox(rle).tolist()
                area = float(coco_mask_utils.area(rle))

                gt_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(gt_labels[j]),
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })

                ann_id += 1

            # -----------------------------------------------------------------
            # Predicted annotations
            # -----------------------------------------------------------------
            if "masks" not in output or len(output["masks"]) == 0:
                continue

            pred_masks = output["masks"].detach().cpu().numpy()    # N x 1 x H x W
            pred_scores = output["scores"].detach().cpu().numpy()
            pred_labels = output["labels"].detach().cpu().numpy()
            pred_boxes = output["boxes"].detach().cpu().numpy()

            for j in range(len(pred_scores)):
                if pred_scores[j] < score_thresh:
                    continue

                # Torchvision Mask R-CNN outputs probability masks.
                binary_mask = (pred_masks[j, 0] > 0.5).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                x1, y1, x2, y2 = pred_boxes[j]

                pred_annotations.append({
                    "image_id": image_id,
                    "category_id": int(pred_labels[j]),
                    "segmentation": rle,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "score": float(pred_scores[j]),
                })

    return images_info, gt_annotations, pred_annotations


# =============================================================================
# Custom fibre-level IoU metric
# =============================================================================

def compute_iou_metrics(gt_anns, pred_anns, iou_thresholds=(0.50, 0.75)):
    """
    Compute ground-truth-centred fibre-level mask IoU metrics.

    For each ground-truth fibre, the predicted mask with the highest IoU
    in the same image is selected. If no prediction exists for that image,
    the ground-truth fibre receives IoU = 0.

    This is not the same as COCO AP. It is a simpler fibre-level distribution
    of best-match mask IoU values, useful for explaining typical mask quality.
    """
    from collections import defaultdict

    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)

    for ann in gt_anns:
        gt_by_image[ann["image_id"]].append(ann)

    for ann in pred_anns:
        pred_by_image[ann["image_id"]].append(ann)

    best_ious = []

    for image_id, image_gt_anns in gt_by_image.items():
        image_pred_anns = pred_by_image.get(image_id, [])

        if not image_pred_anns:
            best_ious.extend([0.0] * len(image_gt_anns))
            continue

        gt_rles = [ann["segmentation"] for ann in image_gt_anns]
        pred_rles = [ann["segmentation"] for ann in image_pred_anns]
        iscrowd = [int(ann.get("iscrowd", 0)) for ann in image_gt_anns]

        # Shape: number of predictions x number of ground-truth objects
        ious = coco_mask_utils.iou(pred_rles, gt_rles, iscrowd)

        for gt_idx in range(len(image_gt_anns)):
            best_iou = float(np.max(ious[:, gt_idx])) if ious.size else 0.0
            best_ious.append(best_iou)

    if len(best_ious) == 0:
        return {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "p5_iou": 0.0,
            "p95_iou": 0.0,
            "n_fibres_iou": 0,
            "fraction_iou_passing_at_0.50": 0.0,
            "fraction_iou_passing_at_0.75": 0.0,
        }

    best_ious = np.asarray(best_ious, dtype=np.float32)

    metrics = {
        "mean_iou": float(np.mean(best_ious)),
        "median_iou": float(np.median(best_ious)),
        "p5_iou": float(np.percentile(best_ious, 5)),
        "p95_iou": float(np.percentile(best_ious, 95)),
        "n_fibres_iou": int(len(best_ious)),
    }

    for threshold in iou_thresholds:
        key = f"fraction_iou_passing_at_{threshold:.2f}"
        metrics[key] = float(np.mean(best_ious >= threshold))

    return metrics


# =============================================================================
# Custom fibre-level coverage metric
# =============================================================================

def compute_coverage_metric(
    gt_anns,
    pred_anns,
    coverage_threshold=0.95,
    precision_floor=0.1,
):
    """
    For each ground-truth fibre, compute:

        coverage = |M_pred ∩ M_gt| / |M_gt|

    using the best-matching prediction that also passes a precision floor:

        |M_pred ∩ M_gt| / |M_pred| >= precision_floor

    A fibre passes if its best-match coverage is >= coverage_threshold.

    The precision floor prevents a giant predicted blob from trivially covering
    every ground-truth fibre. A prediction must have at least `precision_floor`
    of its own area overlapping the ground-truth fibre before it is accepted as
    a valid match.

    Returns:
        {
            "fraction_passing": float,
            "mean_coverage": float,
            "median_coverage": float,
            "p95_coverage": float,
            "p5_coverage": float,
            "n_fibres": int,
        }
    """
    preds_by_image = {}

    for pred in pred_anns:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    pred_areas = {}

    for pred in pred_anns:
        seg = pred["segmentation"]
        rle_enc = {
            "size": seg["size"],
            "counts": seg["counts"].encode("utf-8"),
        }
        pred_areas[id(pred)] = float(coco_mask_utils.area(rle_enc))

    coverages = []

    for gt in gt_anns:
        image_id = gt["image_id"]
        gt_rle = gt["segmentation"]
        gt_area = gt["area"]

        if gt_area == 0:
            continue

        candidate_preds = preds_by_image.get(image_id, [])

        if not candidate_preds:
            coverages.append(0.0)
            continue

        gt_rle_enc = {
            "size": gt_rle["size"],
            "counts": gt_rle["counts"].encode("utf-8"),
        }

        best_coverage = 0.0

        for pred in candidate_preds:
            pred_area = pred_areas[id(pred)]

            if pred_area == 0:
                continue

            pred_seg = pred["segmentation"]
            pred_rle_enc = {
                "size": pred_seg["size"],
                "counts": pred_seg["counts"].encode("utf-8"),
            }

            intersection_rle = coco_mask_utils.merge(
                [gt_rle_enc, pred_rle_enc],
                intersect=True,
            )

            intersection = float(coco_mask_utils.area(intersection_rle))

            # Reject predictions that are mostly outside this ground-truth fibre.
            if intersection / pred_area < precision_floor:
                continue

            coverage = intersection / gt_area

            if coverage > best_coverage:
                best_coverage = coverage

        coverages.append(min(best_coverage, 1.0))

    if len(coverages) == 0:
        return {
            "fraction_passing": 0.0,
            "mean_coverage": 0.0,
            "median_coverage": 0.0,
            "p95_coverage": 0.0,
            "p5_coverage": 0.0,
            "n_fibres": 0,
        }

    coverages = np.asarray(coverages, dtype=np.float32)

    return {
        "fraction_passing": float(np.mean(coverages >= coverage_threshold)),
        "mean_coverage": float(np.mean(coverages)),
        "median_coverage": float(np.median(coverages)),
        "p95_coverage": float(np.percentile(coverages, 95)),
        "p5_coverage": float(np.percentile(coverages, 5)),
        "n_fibres": int(len(coverages)),
    }


# =============================================================================
# COCO evaluation runner
# =============================================================================

def evaluate_coco(
    model,
    loader,
    device,
    score_thresh=0.3,
    coverage_threshold=0.95,
):
    """
    Run COCO-style evaluation and custom fibre-level metrics.

    Returns a dictionary containing:
        AP_mask, AP50_mask, AP75_mask
        AP_box, AP50_box, AP75_box
        coverage metrics
        fibre-level matched IoU metrics
    """
    images_info, gt_anns, pred_anns = collect_predictions(
        model,
        loader,
        device,
        score_thresh,
    )

    if len(gt_anns) == 0:
        print("  Warning: no ground-truth annotations found in this split.")
        return {}

    # -------------------------------------------------------------------------
    # Build COCO ground-truth object in memory
    # -------------------------------------------------------------------------
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images_info,
        "annotations": gt_anns,
        "categories": [{"id": 1, "name": "fibre"}],
    }
    coco_gt.createIndex()

    metrics = {}

    # -------------------------------------------------------------------------
    # Standard COCO AP metrics
    # -------------------------------------------------------------------------
    if len(pred_anns) == 0:
        print("  Warning: no predictions above score threshold.")

        metrics.update({
            "AP_mask": 0.0,
            "AP50_mask": 0.0,
            "AP75_mask": 0.0,
            "AP_box": 0.0,
            "AP50_box": 0.0,
            "AP75_box": 0.0,
        })

    else:
        coco_dt = coco_gt.loadRes(pred_anns)

        for iou_type, prefix in [("segm", "mask"), ("bbox", "box")]:
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats

            metrics[f"AP_{prefix}"] = float(stats[0])
            metrics[f"AP50_{prefix}"] = float(stats[1])
            metrics[f"AP75_{prefix}"] = float(stats[2])

    # -------------------------------------------------------------------------
    # M3-R01 coverage metric
    # -------------------------------------------------------------------------
    coverage_metrics = compute_coverage_metric(
        gt_anns,
        pred_anns,
        coverage_threshold=coverage_threshold,
    )

    metrics.update({
        f"coverage_fraction_passing_at_{coverage_threshold}":
            coverage_metrics["fraction_passing"],
        "coverage_mean": coverage_metrics["mean_coverage"],
        "coverage_median": coverage_metrics["median_coverage"],
        "coverage_p5": coverage_metrics["p5_coverage"],
        "coverage_p95": coverage_metrics["p95_coverage"],
        "n_fibres": coverage_metrics["n_fibres"],
    })

    # -------------------------------------------------------------------------
    # Fibre-level matched IoU metric
    # -------------------------------------------------------------------------
    iou_metrics = compute_iou_metrics(
        gt_anns,
        pred_anns,
        iou_thresholds=(0.50, 0.75),
    )

    metrics.update({
        "iou_mean": iou_metrics["mean_iou"],
        "iou_median": iou_metrics["median_iou"],
        "iou_p5": iou_metrics["p5_iou"],
        "iou_p95": iou_metrics["p95_iou"],
        "iou50_fraction": iou_metrics["fraction_iou_passing_at_0.50"],
        "iou75_fraction": iou_metrics["fraction_iou_passing_at_0.75"],
        "n_fibres_iou": iou_metrics["n_fibres_iou"],
    })

    return metrics