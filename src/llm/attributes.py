"""
Deterministic T3-style attribute extraction from a PREDICTED mask + image.

Reproduces the paper's own rule-based structured-attribute computation (from their
data_analysis.py + build_structured_text_annotations.py) so our diagnostic layer speaks
the exact same vocabulary as the dataset. NO LLM here — every field is computed by a
formula, which is what makes the downstream diagnostic defensible ("these facts are
deterministic, only the reasoning over them is model-generated").

Thresholds are lifted verbatim from the dataset's annotation scripts (see CLAUDE.md).
"""
from __future__ import annotations
import numpy as np


# --- discretization thresholds (verbatim from the paper's annotation code) ---
def _scale_bucket(r):
    if r < 0.01: return "tiny"
    if r < 0.03: return "small"
    if r < 0.10: return "medium"
    if r <= 0.25: return "large"
    return "extensive"

def _polarity_bucket(diff):        # diff = mean_defect - mean_background
    if diff < -5: return "dark"
    if diff > 5:  return "bright"
    return "neutral"

def _saliency_bucket(sal):         # sal = |polarity_diff|
    if sal < 10:  return "low"
    if sal <= 40: return "medium"
    return "high"

def _number_bucket(k):
    return {1:"one",2:"two",3:"three",4:"four",5:"five"}.get(k, "multiple")


def _connected_components(binary):
    """8-connectivity component count + label map, dependency-free (BFS flood fill)."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        for j in range(w):
            if binary[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                stack.append((ny, nx))
    return labels, cur


def _position_grid(binary):
    """3x3 grid; per-cell defect-pixel fraction, ranked, cells with >0 kept."""
    h, w = binary.shape
    names = [["top-left","top-center","top-right"],
             ["middle-left","center","middle-right"],
             ["bottom-left","bottom-center","bottom-right"]]
    total = binary.sum()
    if total == 0:
        return []
    cells = []
    for gi in range(3):
        for gj in range(3):
            ys, ye = gi*h//3, (gi+1)*h//3
            xs, xe = gj*w//3, (gj+1)*w//3
            frac = float(binary[ys:ye, xs:xe].sum()) / float(total)
            if frac > 0:
                cells.append([names[gi][gj], round(frac, 6)])
    cells.sort(key=lambda c: -c[1])
    return cells


def _shape_direction(binary, labels, n_comp):
    """Lightweight geometry -> shape + direction, from the largest component's aspect &
    orientation. (The paper used constrained LLM inference for these two; we derive them
    cheaply from mask geometry so the runtime layer needs no per-image LLM call for facts.)"""
    if n_comp == 0:
        return "unclear", "none"
    # largest component
    sizes = [(labels == k).sum() for k in range(1, n_comp+1)]
    big = 1 + int(np.argmax(sizes))
    ys, xs = np.where(labels == big)
    if len(ys) < 3:
        return "spot-like", "none"
    hspan = ys.max() - ys.min() + 1
    wspan = xs.max() - xs.min() + 1
    ratio = wspan / max(hspan, 1)
    fill = len(ys) / float(hspan * wspan)         # how filled the bbox is
    # direction from aspect
    if ratio > 2.0:   direction = "horizontal"
    elif ratio < 0.5: direction = "vertical"
    else:             direction = "diagonal" if n_comp == 1 else "none"
    # shape heuristic
    if n_comp >= 5:                 shape = "scattered" if fill < 0.3 else "fragmented"
    elif max(ratio, 1/ratio) > 3:   shape = "linear"
    elif fill > 0.7:                shape = "patch-like" if max(hspan,wspan) > 40 else "spot-like"
    else:                           shape = "irregular"
    return shape, direction


def extract_attributes(image_gray: np.ndarray, pred_mask: np.ndarray) -> dict:
    """
    image_gray: (H,W) uint8/float grayscale steel image
    pred_mask:  (H,W) predicted mask (any range) -> binarized at >0.5 (probs) or >127 (uint8)
    Returns the paper's 9-field T3-style tuple, all deterministic.
    """
    img = np.asarray(image_gray).astype(np.float32)
    m = np.asarray(pred_mask)
    binary = (m > 0.5) if m.max() <= 1.0 else (m > 127)
    binary = binary.astype(np.uint8)

    total = int(binary.sum())
    area_ratio = total / float(binary.size)

    if total == 0:
        return {"Defect type": None, "Shape": "none", "Direction": "none",
                "Spatial Distribution": "none", "Number of Defects": "zero",
                "Position": [], "Scale": "none", "Polarity": "neutral",
                "Saliency": "low", "_raw": {"area_ratio": 0.0}}

    mean_defect = float(img[binary == 1].mean())
    mean_bg = float(img[binary == 0].mean()) if (binary == 0).any() else mean_defect
    polarity_diff = mean_defect - mean_bg
    saliency = abs(polarity_diff)

    labels, n_comp = _connected_components(binary)
    shape, direction = _shape_direction(binary, labels, n_comp)
    spatial = ("isolated" if n_comp == 1 else
               "clustered" if n_comp <= 4 else "scattered")

    return {
        "Shape": shape,
        "Direction": direction,
        "Spatial Distribution": spatial,
        "Number of Defects": _number_bucket(n_comp),
        "Position": _position_grid(binary),
        "Scale": _scale_bucket(area_ratio),
        "Polarity": _polarity_bucket(polarity_diff),
        "Saliency": _saliency_bucket(saliency),
        "_raw": {"area_ratio": round(area_ratio, 5),
                 "polarity_diff": round(polarity_diff, 2),
                 "saliency": round(saliency, 2),
                 "n_components": n_comp},
    }
