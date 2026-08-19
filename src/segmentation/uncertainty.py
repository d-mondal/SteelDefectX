"""
Uncertainty quantification + calibration for the segmentation model.

Two capabilities:
  1) MC-Dropout: with dropout kept active at inference, run N stochastic forward passes.
     - mean probability = the prediction
     - std across passes = per-pixel predictive UNCERTAINTY (the heatmap)
     This is the Gal & Ghahramani (2016) approximation to Bayesian inference.
  2) Calibration: Expected Calibration Error (ECE) + reliability-diagram bins, which
     measure whether the model's confidence is HONEST (does "90% sure" mean ~90% correct?).

Why it matters industrially: an inspection system should flag low-confidence predictions
for human review instead of silently trusting every pixel. Uncertainty + calibration are
exactly the signals that enable that.
"""
from __future__ import annotations
import numpy as np
import torch

from src.segmentation.model import enable_mc_dropout


@torch.no_grad()
def mc_dropout_predict(model, images, n_passes: int = 30, device="cuda"):
    """Run N stochastic forward passes with dropout active.
    Returns (mean_prob, uncertainty) each shape (B,1,H,W), on CPU.
      mean_prob   = average sigmoid probability over passes (the prediction)
      uncertainty = std of the probability over passes (predictive uncertainty)
    """
    enable_mc_dropout(model)
    images = images.to(device)
    probs = []
    for _ in range(n_passes):
        p = torch.sigmoid(model(images))
        probs.append(p.cpu())
    stack = torch.stack(probs, dim=0)          # (N,B,1,H,W)
    return stack.mean(0), stack.std(0)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Expected Calibration Error over pixels (pure numpy, GPU-free, unit-testable).
      probs:  predicted defect probability in [0,1], any shape
      labels: ground-truth in {0,1}, same shape
    Returns (ece, bins) where bins is a list of per-bin (conf, acc, weight) for the
    reliability diagram. ECE = sum over bins of weight * |acc - conf|.
    """
    p = probs.reshape(-1).astype(np.float64)
    y = labels.reshape(-1).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so prob==1.0 is included
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        w = int(mask.sum())
        if w == 0:
            bins.append({"bin": (lo, hi), "conf": None, "acc": None, "weight": 0})
            continue
        conf = float(p[mask].mean())          # avg predicted confidence in bin
        acc = float(y[mask].mean())           # actual fraction of positives in bin
        ece += (w / n) * abs(acc - conf)
        bins.append({"bin": (lo, hi), "conf": conf, "acc": acc, "weight": w})
    return float(ece), bins


@torch.no_grad()
def evaluate_calibration(model, loader, device="cuda", n_bins: int = 15,
                         mc_passes: int = 0):
    """Stream the test set, gather (prob, label) pixels, return ECE + bins + mean uncertainty.
    If mc_passes>0, uses MC-Dropout mean prob and also returns mean per-pixel uncertainty;
    otherwise a single deterministic forward pass (standard calibration).
    """
    all_conf, all_lab = [], []
    unc_sum, unc_count = 0.0, 0
    for images, masks, _ in loader:
        if mc_passes > 0:
            mean_p, unc = mc_dropout_predict(model, images, mc_passes, device)
            unc_sum += float(unc.sum()); unc_count += unc.numel()
        else:
            model.eval()
            mean_p = torch.sigmoid(model(images.to(device))).cpu()
        all_conf.append(mean_p.reshape(-1).numpy())
        all_lab.append(masks.reshape(-1).numpy())
    conf = np.concatenate(all_conf)
    lab = np.concatenate(all_lab)
    ece, bins = compute_ece(conf, lab, n_bins)
    out = {"ECE": round(ece * 100, 3), "bins": bins}
    if mc_passes > 0 and unc_count:
        out["mean_uncertainty"] = round(unc_sum / unc_count, 5)
    return out
