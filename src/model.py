"""
model.py -- Mask R-CNN for fibre instance segmentation
======================================================
Wraps torchvision's Mask R-CNN with a ResNet-50-FPN backbone,
pre-trained on COCO, and replaces the heads for a 2-class problem
(background + fibre).

Two variants:
  build_model("maskrcnn_resnet50_fpn")    -- ResNet-50 backbone  (fast, good)
  build_model("maskrcnn_resnet50_fpn_v2") -- ResNet-50 V2 backbone (better accuracy)
"""

import torchvision
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


NUM_CLASSES = 2   # 0 = background, 1 = fibre


def build_model(variant="maskrcnn_resnet50_fpn_v2", pretrained=True, trainable_backbone_layers=3):
    """
    Build a Mask R-CNN model fine-tuned for fibre segmentation.

    Args:
        variant                  : torchvision model name
        pretrained               : load COCO pre-trained weights
        trainable_backbone_layers: how many FPN layers to unfreeze (0-5)
                                   3 is a good default -- keeps early layers frozen

    Returns:
        model: torch.nn.Module ready for training
    """
    if variant == "maskrcnn_resnet50_fpn_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model   = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model   = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )

    # -- Replace box classifier head --
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, NUM_CLASSES)

    # -- Replace mask head --
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_CLASSES
    )

    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
