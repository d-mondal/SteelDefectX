"""
Create a FROZEN, class-stratified train / internal-val split for SteelDefectX.

Design (locked project decisions):
  - The dataset's official `val/` split is our HELD-OUT TEST set. We never tune on it,
    so the final IoU-vs-paper number stays honest. This script does NOT touch it.
  - We carve an INTERNAL VALIDATION set out of the official `train/` split, for
    checkpoint selection / early stopping.
  - The carve is STRATIFIED BY CLASS. With a long tail (Rolled pit=35, Oxide scale of
    plate system=45, Crease=53) a naive random split could leave a rare class with ~0
    val images and wreck its per-class signal. Per-class splitting guarantees every
    class appears in both train and internal-val.
  - Frozen once, saved to disk as plain id lists. Every architecture and all 3 seeds
    train on the IDENTICAL split, so the 3-seed spread measures MODEL variance, not
    split variance. This is what makes the headline comparison defensible.

Run:
    python -m src.data.make_split \
        --train-text /path/to/train-text.json \
        --out-dir data/splits \
        --val-frac 0.15 --seed 42
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load_entries(train_text_path: Path):
    """Return list of (image_name, class_name) from the dataset's train-text.json."""
    with open(train_text_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else list(data.values())

    # robustly find the two fields we need (schema confirmed in the Day-0 audit,
    # but stay defensive in case a future dataset version renames keys)
    e0 = entries[0]
    name_key = next((k for k in e0 if "image" in k.lower() and "name" in k.lower()), "image_name")
    class_key = next((k for k in e0 if k.lower() in ("class_name", "class", "classname", "label")), "class_name")
    return [(e[name_key], e[class_key]) for e in entries], name_key, class_key


def stratified_split(items, val_frac: float, seed: int):
    """
    items: list of (id, class_label).
    Returns (train_ids, val_ids). Every class is split independently so each class
    contributes ~val_frac of its own images to internal-val (>=1 if the class has >=2).
    """
    by_class = defaultdict(list)
    for _id, cls in items:
        by_class[cls].append(_id)

    rng = random.Random(seed)
    train_ids, val_ids = [], []
    per_class_report = {}
    for cls in sorted(by_class):
        ids = by_class[cls][:]
        rng.shuffle(ids)
        n = len(ids)
        # at least 1 to val if the class has >=2 images; never take the whole class
        n_val = min(max(1, math.floor(val_frac * n)), n - 1) if n >= 2 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
        per_class_report[cls] = {"total": n, "train": n - n_val, "val": n_val}
    return train_ids, val_ids, per_class_report


def write_ids(path: Path, ids):
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-text", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/splits"), type=Path)
    ap.add_argument("--val-frac", default=0.15, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items, name_key, class_key = load_entries(args.train_text)
    train_ids, val_ids, report = stratified_split(items, args.val_frac, args.seed)

    write_ids(args.out_dir / "train_ids.txt", train_ids)
    write_ids(args.out_dir / "internalval_ids.txt", val_ids)

    meta = {
        "source": str(args.train_text),
        "name_key": name_key, "class_key": class_key,
        "val_frac": args.val_frac, "seed": args.seed,
        "n_train": len(train_ids), "n_internalval": len(val_ids),
        "n_total_train_pool": len(items),
        "note": "Official val/ split is the held-out TEST set; not represented here.",
        "per_class": report,
    }
    (args.out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train={len(train_ids)}  internal-val={len(val_ids)}  "
          f"(from {len(items)} train-pool images, val_frac={args.val_frac}, seed={args.seed})")
    print(f"wrote: {args.out_dir}/train_ids.txt, internalval_ids.txt, split_meta.json")
    # flag any class whose internal-val allocation is tiny (worth eyeballing)
    thin = {c: r for c, r in report.items() if r["val"] <= 5}
    if thin:
        print("thin internal-val classes (val<=5):")
        for c, r in sorted(thin.items(), key=lambda x: x[1]["val"]):
            print(f"  {c:32s} total={r['total']:4d} train={r['train']:4d} val={r['val']:2d}")


if __name__ == "__main__":
    main()
