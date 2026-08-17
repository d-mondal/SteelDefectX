"""
Segmentation metrics.

Two jobs:
  1) Report the SAME three metrics the paper's Table 3 uses (AUROC, F1-max, IoU),
     so "first supervised baseline vs paper zero-shot (37.49 IoU)" is apples-to-apples
     on the metric definitions. These are pixel-level, micro-averaged over the test set.
  2) Report PER-CLASS IoU/Dice (macro over images within each class) — a new result the
     paper never provides, and the thing that shows all 25 classes incl. rare ones.

Memory note: pixel-level AUROC / F1-max over the whole test set would be billions of
values if materialised. We instead accumulate a per-threshold confusion (TP/FP/FN/TN)
over a fixed grid of thresholds in a single streaming pass — O(n_thresholds) memory.
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch


class SegMetricAccumulator:
    """Streaming accumulator. Call update() per batch, then compute()."""

    def __init__(self, n_thresholds: int = 50, iou_threshold: float = 0.5):
        self.thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)[1:-1]  # exclude 0,1
        self.iou_threshold = iou_threshold
        # per-threshold pixel confusion, micro over dataset
        self.tp = np.zeros_like(self.thresholds, dtype=np.int64)
        self.fp = np.zeros_like(self.thresholds, dtype=np.int64)
        self.fn = np.zeros_like(self.thresholds, dtype=np.int64)
        self.tn = np.zeros_like(self.thresholds, dtype=np.int64)
        # micro IoU/Dice at the fixed operating point
        self._inter = 0
        self._union = 0
        self._pred_sum = 0
        self._gt_sum = 0
        # per-class: list of per-image IoU / Dice at fixed threshold
        self.per_class_iou = defaultdict(list)
        self.per_class_dice = defaultdict(list)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor, classes):
        """
        probs:   (B,1,H,W) sigmoid probabilities in [0,1]
        targets: (B,1,H,W) in {0,1}
        classes: length-B iterable of class-name strings
        """
        p = probs.detach().float().cpu().numpy().reshape(probs.shape[0], -1)
        g = targets.detach().float().cpu().numpy().reshape(targets.shape[0], -1).astype(np.int64)

        # --- per-threshold confusion (micro over all pixels in this batch) ---
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        for i, t in enumerate(self.thresholds):
            pred = flat_p >= t
            self.tp[i] += int(np.count_nonzero(pred & (flat_g == 1)))
            self.fp[i] += int(np.count_nonzero(pred & (flat_g == 0)))
            self.fn[i] += int(np.count_nonzero((~pred) & (flat_g == 1)))
            self.tn[i] += int(np.count_nonzero((~pred) & (flat_g == 0)))

        # --- fixed-threshold micro IoU/Dice + per-image per-class ---
        pred_fixed = (p >= self.iou_threshold).astype(np.int64)
        for b in range(p.shape[0]):
            pr, gt = pred_fixed[b], g[b]
            inter = int(np.count_nonzero((pr == 1) & (gt == 1)))
            psum, gsum = int(pr.sum()), int(gt.sum())
            union = psum + gsum - inter
            self._inter += inter
            self._union += union
            self._pred_sum += psum
            self._gt_sum += gsum
            # per-image IoU/Dice (define empty-empty as 1.0: correctly found nothing)
            iou = 1.0 if union == 0 else inter / union
            dice = 1.0 if (psum + gsum) == 0 else (2 * inter) / (psum + gsum)
            self.per_class_iou[classes[b]].append(iou)
            self.per_class_dice[classes[b]].append(dice)

    def compute(self):
        # AUROC via trapezoidal integration of the ROC (TPR vs FPR) over thresholds
        tpr = self.tp / np.maximum(self.tp + self.fn, 1)
        fpr = self.fp / np.maximum(self.fp + self.tn, 1)
        # sort by fpr ascending, prepend (0,0)/append (1,1) endpoints
        order = np.argsort(fpr)
        fpr_s, tpr_s = fpr[order], tpr[order]
        fpr_s = np.concatenate([[0.0], fpr_s, [1.0]])
        tpr_s = np.concatenate([[0.0], tpr_s, [1.0]])
        _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np2.0 renamed trapz
        auroc = float(_trap(tpr_s, fpr_s))

        # F1 across thresholds -> F1-max
        precision = self.tp / np.maximum(self.tp + self.fp, 1)
        recall = self.tp / np.maximum(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        f1_max = float(f1.max())

        micro_iou = 1.0 if self._union == 0 else self._inter / self._union
        micro_dice = 1.0 if (self._pred_sum + self._gt_sum) == 0 else \
            (2 * self._inter) / (self._pred_sum + self._gt_sum)

        per_class = {
            c: {"IoU": float(np.mean(self.per_class_iou[c])),
                "Dice": float(np.mean(self.per_class_dice[c])),
                "n": len(self.per_class_iou[c])}
            for c in sorted(self.per_class_iou)
        }
        macro_iou = float(np.mean([v["IoU"] for v in per_class.values()]))

        return {
            # reported on 0-100 scale to line up with the paper's Table 3
            "AUROC": round(auroc * 100, 2),
            "F1max": round(f1_max * 100, 2),
            "IoU": round(micro_iou * 100, 2),        # micro, the headline vs 37.49
            "Dice": round(micro_dice * 100, 2),
            "IoU_macro": round(macro_iou * 100, 2),  # class-balanced view
            "per_class": per_class,
        }
