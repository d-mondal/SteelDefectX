"""
Dice + BCE loss for binary defect segmentation.

Why both: BCE gives stable per-pixel gradients; Dice directly optimises overlap and
copes with the heavy background-vs-defect imbalance inside each 256x256 image (most
pixels are background). The weighted sum is a standard, well-understood combination.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets_f = targets.reshape(targets.size(0), -1)
        inter = (probs * targets_f).sum(1)
        dice = 1 - (2 * inter + self.smooth) / (probs.sum(1) + targets_f.sum(1) + self.smooth)
        dice = dice.mean()
        return self.dice_weight * dice + (1 - self.dice_weight) * bce
