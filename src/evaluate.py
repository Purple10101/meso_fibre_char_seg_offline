"""
evaluate.py -- COCO-style evaluation for fibre instance segmentation
====================================================================
Returns AP (averaged over IoU thresholds) for both masks and boxes,
matching the standard COCO evaluation protocol used by detectron2, MMDet, etc.
"""

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils


# --- Prediction collector -----------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device, score_thresh=0.3):
    """
    Run inference over a DataLoader and collect predictions + ground truths
    in COCO-annotation format for pycocotools evaluation.
    """
    model.eval()

    gt_annotations   = []
    pred_annotations = []
    images_info      = []
    ann_id = 1

    for batch_idx, (images, targets) in enumerate(loader):
        images_gpu = [img.to(device) for img in images]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images_gpu)

        for i, (output, target) in enumerate(zip(outputs, targets)):
            image_id = int(target["image_id"].item())
            H, W     = images[i].shape[-2], images[i].shape[-1]

            images_info.append({"id": image_id, "height": H, "width": W})

            # -- Ground truth --
            gt_masks  = target["masks"].cpu().numpy()   # NxHxW uint8
            gt_labels = target["labels"].cpu().numpy()  # N

            for j in range(len(gt_masks)):
                rle = coco_mask_utils.encode(
                    np.asfortranarray(gt_masks[j])
                )
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox = coco_mask_utils.toBbox(rle).tolist()
                area = float(coco_mask_utils.area(rle))
                gt_annotations.append({
                    "id":           ann_id,
                    "image_id":     image_id,
                    "category_id":  int(gt_labels[j]),
                    "segmentation": rle,
                    "bbox":         bbox,
                    "area":         area,
                    "iscrowd":      0,
                })
                ann_id += 1

            # -- Predictions --
            pred_masks  = output["masks"].cpu().numpy()   # Nx1xHxW float
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_boxes  = output["boxes"].cpu().numpy()

            for j in range(len(pred_scores)):
                if pred_scores[j] < score_thresh:
                    continue
                # threshold mask at 0.5
                binary_mask = (pred_masks[j, 0] > 0.5).astype(np.uint8)
                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                pred_annotations.append({
                    "image_id":     image_id,
                    "category_id":  int(pred_labels[j]),
                    "segmentation": rle,
                    "bbox": [
                                float(pred_boxes[j][0]),
                                float(pred_boxes[j][1]),
                                float(pred_boxes[j][2] - pred_boxes[j][0]),
                                float(pred_boxes[j][3] - pred_boxes[j][1]),
                            ],
                    "score":        float(pred_scores[j]),
                })

    return images_info, gt_annotations, pred_annotations


# --- COCO eval runner ---------------------------------------------------------

def evaluate_coco(model, loader, device, score_thresh=0.3, coverage_threshold=0.95):
    images_info, gt_anns, pred_anns = collect_predictions(
        model, loader, device, score_thresh
    )

    if len(gt_anns) == 0:
        print("  Warning: no ground-truth annotations found in this split.")
        return {}

    # -- Build COCO GT object in-memory --
    coco_gt = COCO()
    coco_gt.dataset = {
        "images":     images_info,
        "annotations": gt_anns,
        "categories":  [{"id": 1, "name": "fibre"}],
    }
    coco_gt.createIndex()

    metrics = {}

    if len(pred_anns) == 0:
        print("  Warning: no predictions above score threshold.")
        metrics = {"AP_mask": 0.0, "AP50_mask": 0.0, "AP75_mask": 0.0,
                   "AP_box":  0.0, "AP50_box":  0.0, "AP75_box":  0.0}
    else:
        coco_dt = coco_gt.loadRes(pred_anns)

        for iou_type, prefix in [("segm", "mask"), ("bbox", "box")]:
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            metrics[f"AP_{prefix}"]   = float(stats[0])
            metrics[f"AP50_{prefix}"] = float(stats[1])
            metrics[f"AP75_{prefix}"] = float(stats[2])

    # -- M3-R01 coverage metric --
    coverage_metrics = compute_coverage_metric(gt_anns, pred_anns, coverage_threshold)
    metrics.update({
        f"coverage_fraction_passing_at_{coverage_threshold}": coverage_metrics["fraction_passing"],
        "coverage_mean":   coverage_metrics["mean_coverage"],
        "coverage_median": coverage_metrics["median_coverage"],
        "coverage_p5":     coverage_metrics["p5_coverage"],
        "coverage_p95":    coverage_metrics["p95_coverage"],
        "n_fibres":        coverage_metrics["n_fibres"],
    })

    return metrics


# --- Add this new function ----------------------------------------------------

def compute_coverage_metric(gt_anns, pred_anns, coverage_threshold=0.95,
                            precision_floor=0.1):
    """
    For each GT fibre, compute coverage = |M_pred n M_gt| / |M_gt| using the
    best-matching prediction that also passes a precision floor:
        |M_pred n M_gt| / |M_pred| >= precision_floor

    A fibre passes if its best-match coverage >= coverage_threshold.

    The precision floor prevents a giant blob from trivially covering every
    GT fibre -- a prediction must have at least `precision_floor` of its own
    area overlapping the GT before it is accepted as a valid match.

    Returns:
        {
          "fraction_passing": float,   # fraction of GT fibres with coverage >= threshold
          "mean_coverage":    float,   # mean coverage across all GT fibres
          "median_coverage":  float,
          "p95_coverage":     float,   # 95th percentile
          "p5_coverage":      float,   # 5th percentile
          "n_fibres":         int,
        }
    """
    # Group predictions by image for efficient lookup
    preds_by_image = {}
    for p in pred_anns:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    # Pre-compute prediction areas (needed for precision floor check)
    pred_areas = {}
    for p in pred_anns:
        seg = p["segmentation"]
        rle_enc = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
        pred_areas[id(p)] = float(coco_mask_utils.area(rle_enc))

    coverages = []

    for gt in gt_anns:
        image_id = gt["image_id"]
        gt_rle   = gt["segmentation"]
        gt_area  = gt["area"]

        if gt_area == 0:
            continue

        candidate_preds = preds_by_image.get(image_id, [])

        if not candidate_preds:
            coverages.append(0.0)
            continue

        gt_rle_enc = {"size": gt_rle["size"], "counts": gt_rle["counts"].encode("utf-8")}

        best_coverage = 0.0
        for p in candidate_preds:
            pred_area = pred_areas[id(p)]
            if pred_area == 0:
                continue

            p_rle_enc    = {"size": p["segmentation"]["size"],
                            "counts": p["segmentation"]["counts"].encode("utf-8")}
            merged       = coco_mask_utils.merge([gt_rle_enc, p_rle_enc], intersect=True)
            intersection = float(coco_mask_utils.area(merged))

            # Reject predictions that are mostly outside this GT
            if intersection / pred_area < precision_floor:
                continue

            coverage = intersection / gt_area
            if coverage > best_coverage:
                best_coverage = coverage

        coverages.append(min(best_coverage, 1.0))

    coverages = np.array(coverages)

    if len(coverages) == 0:
        return {"fraction_passing": 0.0, "mean_coverage": 0.0,
                "median_coverage": 0.0, "p95_coverage": 0.0,
                "p5_coverage": 0.0, "n_fibres": 0}

    return {
        "fraction_passing": float((coverages >= coverage_threshold).mean()),
        "mean_coverage": float(coverages.mean()),
        "median_coverage": float(np.median(coverages)),
        "p95_coverage": float(np.percentile(coverages, 95)),
        "p5_coverage": float(np.percentile(coverages, 5)),
        "n_fibres": int(len(coverages)),
    }