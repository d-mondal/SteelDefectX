"""
Aggregate all segmentation run results into publication-ready tables.

Reads every *_test_metrics.json in a results dir, groups by architecture, and prints:
  1) Head-to-head: per-arch mean +/- std (IoU/F1max/AUROC) vs the paper's zero-shot ceiling.
  2) Per-class IoU averaged across seeds, per arch, sorted worst-first.

Works with a partial sweep (e.g. only U-Net done) and fills in as more runs land.
Run standalone:  python -m src.evaluation.aggregate --results-dir results/segmentation
Or import aggregate(results_dir) from a notebook.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PAPER_ZEROSHOT = {"AUROC": 88.21, "F1max": 53.36, "IoU": 37.49}


def _arch_of(fname: str) -> str:
    base = os.path.basename(fname)
    # names look like: unet_resnet34_seed0_test_metrics.json  /  deeplabv3plus_resnet34_seed2_...
    m = re.match(r"([a-zA-Z0-9]+)_", base)
    return m.group(1) if m else "unknown"


def aggregate(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "*_test_metrics.json")))
    if not files:
        print(f"no result files in {results_dir}")
        return {}

    by_arch = defaultdict(list)
    for f in files:
        by_arch[_arch_of(f)].append(json.load(open(f)))

    archs = sorted(by_arch)
    headline = ("IoU", "F1max", "AUROC", "Dice", "IoU_macro")

    # ---- 1) head-to-head table ----
    agg = {}
    for a in archs:
        runs = by_arch[a]
        agg[a] = {"n_seeds": len(runs)}
        for k in headline:
            vals = [r[k] for r in runs if k in r]
            agg[a][k] = (round(float(np.mean(vals)), 2),
                         round(float(np.std(vals)), 2)) if vals else (float("nan"), 0.0)

    print("=" * 78)
    print("HEAD-TO-HEAD: supervised (ours) vs paper zero-shot")
    print("Framing: FIRST supervised baseline vs the paper's zero-shot ceiling.")
    print("=" * 78)
    header = f"{'metric':10s} | {'paper 0-shot':>12s} | " + " | ".join(
        f"{a} (n={agg[a]['n_seeds']})".rjust(22) for a in archs)
    print(header)
    print("-" * len(header))
    for k in ("IoU", "F1max", "AUROC"):
        row = f"{k:10s} | {PAPER_ZEROSHOT.get(k, float('nan')):>12.2f} | "
        row += " | ".join(f"{agg[a][k][0]:>10.2f} +/- {agg[a][k][1]:<7.2f}" for a in archs)
        print(row)
    # improvement line for the headline metric
    for a in archs:
        delta = agg[a]["IoU"][0] - PAPER_ZEROSHOT["IoU"]
        print(f"  -> {a}: IoU {agg[a]['IoU'][0]:.2f} vs 37.49  = +{delta:.2f} pts over paper zero-shot")

    # ---- 2) per-class IoU averaged across seeds ----
    print("\n" + "=" * 78)
    print("PER-CLASS IoU (mean across seeds) — the breakdown the paper never reports")
    print("=" * 78)
    for a in archs:
        runs = by_arch[a]
        cls_iou = defaultdict(list)
        cls_n = {}
        for r in runs:
            for c, v in r.get("per_class", {}).items():
                cls_iou[c].append(v["IoU"] * 100)
                cls_n[c] = v["n"]
        rows = sorted(((c, float(np.mean(v)), float(np.std(v)), cls_n[c])
                       for c, v in cls_iou.items()), key=lambda x: x[1])
        print(f"\n[{a}]  (worst-first)")
        for c, mu, sd, n in rows:
            print(f"  {c:34s} IoU={mu:5.1f} +/- {sd:4.1f}   (n={n})")

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/segmentation")
    args = ap.parse_args()
    aggregate(args.results_dir)


if __name__ == "__main__":
    main()
