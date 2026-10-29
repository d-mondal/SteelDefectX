"""
Lightweight 25-class defect CLASSIFIER + unknown-class check.

Purpose: auto-assign the defect class for an uploaded image so the dashboard's
diagnostic pipeline runs end-to-end without the user picking a class by hand.

Design: a ResNet34 (ImageNet-pretrained) backbone + a linear head over 25 classes.
Same backbone family as the segmentation encoder, so it's a consistent, defensible
choice. Trained quickly (a few epochs) on the SteelDefectX train split; the paper
already showed classification is easy on this data (~94%), so this is a sanity-grade
auxiliary model, not a headline contribution.

Unknown-class check: a softmax classifier always picks one of its 25 classes, even for
a defect it has never seen. We therefore reject a prediction when its top-class
probability falls below a threshold and label the image "requires manual class
labelling". The threshold is chosen on the INTERNAL VALIDATION split (never the test
split) so that a target share of correctly classified known-class images is still
accepted (maximum-softmax-probability rule). Note what this can and cannot do: it
catches images the classifier is unsure about; an unseen defect that the classifier is
confidently wrong about will still pass. We have no images of unseen classes to measure
that, so report the check as a confidence gate, not as a proven new-class detector.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

UNKNOWN_LABEL = "Unknown class (requires manual class labelling)"


def build_classifier(num_classes: int = 25, pretrained: bool = True):
    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet34(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


@torch.no_grad()
def predict_class(net, gray_img, class_names, device="cpu"):
    """gray_img: (H,W) uint8/float grayscale. Returns (class_name, confidence, full_probs)."""
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


# ------------------------- unknown-class check -------------------------
def choose_unknown_threshold(top_probs, correct, target_accept: float = 0.95) -> float:
    """Pick the rejection threshold from INTERNAL-VAL predictions.

    top_probs: top-class softmax probability per internal-val image.
    correct:   bool per image, whether the top class was right.
    Returns the largest threshold that still accepts `target_accept` of the correctly
    classified images (the (1 - target_accept) quantile of their top probability).
    """
    top_probs = np.asarray(top_probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if not correct.any():
        raise ValueError("no correctly classified images to calibrate the threshold on")
    return float(np.quantile(top_probs[correct], 1.0 - target_accept))


def apply_unknown_threshold(class_name: str, confidence: float, threshold: float | None):
    """Returns (label, is_unknown). With no threshold available, nothing is rejected."""
    if threshold is not None and confidence < threshold:
        return UNKNOWN_LABEL, True
    return class_name, False


def rejection_report(top_probs, correct, threshold: float) -> dict:
    """What the threshold does on a labelled split (all images are known classes here)."""
    top_probs = np.asarray(top_probs, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    accepted = top_probs >= threshold
    n = len(top_probs)
    n_wrong = int((~correct).sum())
    return {
        "n": n,
        "threshold": round(float(threshold), 4),
        "accepted_pct": round(float(100 * accepted.mean()), 2),
        "rejected_n": int((~accepted).sum()),
        "accuracy_on_accepted_pct": round(float(100 * correct[accepted].mean()), 2) if accepted.any() else None,
        "correct_rejected_pct": round(float(100 * (~accepted & correct).sum() / max(correct.sum(), 1)), 2),
        "errors_caught_pct": round(float(100 * (~accepted & ~correct).sum() / n_wrong), 2) if n_wrong else None,
    }


def save_unknown_threshold(path, threshold: float, target_accept: float, n_internal_val: int,
                           extra: dict | None = None):
    rec = {
        "threshold": round(float(threshold), 6),
        "rule": "reject if top-class softmax probability < threshold",
        "calibrated_on": "internal validation split (data/splits/internalval_ids.txt)",
        "target_accept_of_correct": target_accept,
        "n_internal_val": n_internal_val,
    }
    if extra:
        rec.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def load_unknown_threshold(path) -> float | None:
    """Threshold from notebook 05, or None if the classifier has not been re-trained yet."""
    p = Path(path)
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
