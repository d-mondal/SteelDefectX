"""
Lightweight 25-class defect CLASSIFIER.

Purpose: auto-assign the defect class for an uploaded image so the dashboard's
diagnostic pipeline runs end-to-end without the user picking a class by hand.

Design: a ResNet34 (ImageNet-pretrained) backbone + a linear head over 25 classes.
Same backbone family as the segmentation encoder, so it's a consistent, defensible
choice. Trained quickly (a few epochs) on the SteelDefectX train split; the paper
already showed classification is easy on this data (~94%), so this is a sanity-grade
auxiliary model, not a headline contribution.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision


def build_classifier(num_classes: int = 25, pretrained: bool = True):
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet34(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


@torch.no_grad()
def predict_class(net, gray_img, class_names, device="cpu"):
    """gray_img: (H,W) uint8/float grayscale. Returns (class_name, confidence, full_probs)."""
    import numpy as np
    g = gray_img.astype("float32") / 255.0
    x = np.stack([g, g, g], 0)[None]
    mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225])[None, :, None, None]
    x = torch.tensor((x - mean) / std, dtype=torch.float32).to(device)
    net.eval()
    logits = net(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx]), probs
