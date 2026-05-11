# Mesoscale Fibre Characterisation — Instance Segmentation

Instance segmentation pipeline for detecting and delineating individual fibres in mesoscale images. Benchmarks three architectures (Mask R-CNN, YOLOv8-seg, SOLOv2) against a synthetic fibre dataset and produces Table 21 of the associated paper.

---

## Project structure

```
meso_fibre_char_seg_offline/
├── src/
│   ├── model.py                  # Mask R-CNN (R50-FPN / V2) builder
│   ├── dataset.py                # FibreDataset + albumentations augmentations
│   ├── evaluate.py               # COCO AP + custom coverage & IoU metrics
│   └── benchmark_architectures.py# Runs all three architectures, outputs Table 21
├── fibre_dataset/
│   ├── train/
│   │   ├── images/               # Synthetic RGB fibre images
│   │   └── masks/                # Per-instance colour-coded RGB masks
│   ├── yolo_format/              # YOLO-compatible layout (images/train, labels/train)
│   └── manifest.json             # Colour→instance lookup + train/val/test splits
├── runs/
│   └── fibre_maskrcnn/           # TensorBoard event files from training runs
├── benchmark_results/
│   ├── table21_results.json
│   ├── table21_results.csv
│   ├── table21_latex.txt
│   └── prediction_comparison.png
├── yolov8m-seg.pt                # YOLOv8-seg COCO pretrained weights
└── yolo26n.pt                    # YOLO nano weights
```

---

## Dataset format

The dataset uses **colour-coded instance masks**: each fibre instance is painted a unique RGB colour in a companion mask PNG. `manifest.json` stores the mapping from colour to instance and assigns each sample to a split:

```json
{
  "samples": [
    {
      "image": "train/images/fibre_0000.png",
      "mask":  "train/masks/fibre_0000_mask.png",
      "split": "train",
      "fibres": [
        {"mask_rgb": [128, 64, 32]},
        ...
      ]
    }
  ]
}
```

`manifest.json` is required for training and evaluation — without it the dataset loader will raise an error.

---

## Installation

```bash
pip install torch torchvision pycocotools albumentations   # Mask R-CNN (required)
pip install ultralytics                                     # YOLOv8-seg (optional)
pip install mmdet mmcv mmengine                             # SOLOv2 (optional)
```

You only need to install the frameworks for the architectures you want to benchmark. The benchmark script gracefully skips any whose dependencies are missing.

---

## Running the benchmark

```bash
cd src

# Benchmark all three architectures using existing checkpoints
python benchmark_architectures.py \
    --data_dir ../fibre_dataset \
    --out_dir  ../benchmark_results \
    --maskrcnn_ckpt ../runs/fibre_maskrcnn/best.pth \
    --yolov8_ckpt   ../runs/yolov8_seg/weights/best.pt \
    --solov2_ckpt   ../runs/solov2/best.pth

# Benchmark a subset of models
python benchmark_architectures.py --models maskrcnn yolov8

# Run without fine-tuned checkpoints (uses COCO-pretrained weights — AP will not reflect fibre performance)
python benchmark_architectures.py --models maskrcnn
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `../fibre_dataset` | Root of the fibre dataset (must contain `manifest.json`) |
| `--out_dir` | `../benchmark_results` | Output directory for results |
| `--image_size` | `512` | Input image resolution |
| `--num_workers` | `4` | DataLoader workers |
| `--models` | `maskrcnn yolov8 solov2` | Architectures to run |
| `--maskrcnn_ckpt` | `runs/fibre_maskrcnn/best.pth` | Mask R-CNN checkpoint |
| `--yolov8_ckpt` | `runs/yolov8_seg/weights/best.pt` | YOLOv8-seg checkpoint |
| `--solov2_ckpt` | `None` | SOLOv2 checkpoint |
| `--solov2_config` | `None` | SOLOv2 mmdet config path |

### Outputs

```
benchmark_results/
├── table21_results.json    # Raw metrics for all models
├── table21_results.csv     # CSV version
├── table21_latex.txt       # LaTeX table ready for paper
└── inference_timing.json   # Per-image timing stats
```

---

## Evaluation metrics

### COCO AP
Standard COCO average precision over IoU thresholds 0.5–0.95, computed separately for masks (`segm`) and bounding boxes (`bbox`):
- `AP_mask`, `AP50_mask`, `AP75_mask`
- `AP_box`, `AP50_box`, `AP75_box`

### Coverage metric (M3-R01)
For each ground-truth fibre, the best-matching prediction is found using:

```
coverage = |M_pred ∩ M_gt| / |M_gt|
```

A precision floor (`10%` by default) is applied so that a single large blob cannot trivially cover every ground-truth fibre. A fibre **passes** if its best-match coverage ≥ 95%. The reported metric is the fraction of fibres that pass.

### Fibre-level IoU
For each ground-truth fibre, the prediction with the highest mask IoU in the same image is selected. Reported as a distribution: mean, median, p5, p95, and fraction passing at IoU ≥ 0.50 / 0.75 / 0.95.

---

## Results (Table 21)

| Architecture | AP\_mask | AP50\_mask | Coverage ≥ 95% | IoU mean | IoU median | Inference (ms) | Parameters |
|---|---|---|---|---|---|---|---|
| Mask R-CNN (R50-FPN-V2) | 0.8254 | 0.9883 | 78.8% | 0.904 | 0.917 | 93.7 | 45.9 M |
| YOLOv8-seg | 0.8022 | 0.9950 | 91.8% | 0.886 | 0.896 | 20.0 | 27.2 M |

Mask R-CNN was evaluated at epoch 42. YOLOv8-seg achieves higher coverage and ~4.7× faster inference at the cost of ~2.3 pp AP\_mask. SOLOv2 results pending.

---

## Code overview

### `src/model.py`
Builds a torchvision Mask R-CNN with a ResNet-50-FPN backbone pre-trained on COCO, with the classification and mask heads replaced for the 2-class (background + fibre) problem.

```python
from model import build_model

model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=True, trainable_backbone_layers=3)
```

### `src/dataset.py`
`FibreDataset` decodes per-instance masks from the RGB colour lookup in `manifest.json` and returns targets in the format expected by torchvision's detection collate pipeline. Includes separate albumentations pipelines for train (crop, flip, colour jitter, noise) and val (resize + normalise).

```python
from dataset import FibreDataset, collate_fn, get_train_transforms

ds = FibreDataset("fibre_dataset", split="train", transforms=get_train_transforms(512))
```

### `src/evaluate.py`
Runs a full COCO-style evaluation pass over a DataLoader and computes all metrics described above. The coverage and IoU functions can also be called independently on pre-collected annotation dicts.

```python
from evaluate import evaluate_coco

metrics = evaluate_coco(model, val_loader, device, score_thresh=0.3, coverage_threshold=0.95)
```

### `src/benchmark_architectures.py`
Orchestrates the benchmark: loads each architecture, runs `evaluate_coco` (or the ultralytics/mmdet equivalent), measures median per-image inference latency, and writes results as JSON, CSV, and LaTeX.