"""
Train + evaluate supervised binary segmentation on SteelDefectX.

`train_one_run(cfg, arch, seed)` trains a single model with early stopping on
internal-val IoU, then evaluates once on the held-out TEST set (official val split).

`run_comparison(cfg)` loops {unet, deeplabv3plus} x seeds and aggregates mean +/- std,
then prints the head-to-head table alongside the paper's zero-shot numbers.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import make_datasets
from src.segmentation.model import build_model
from src.segmentation.losses import DiceBCELoss
from src.segmentation.metrics import SegMetricAccumulator


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=True, drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    acc = SegMetricAccumulator(n_thresholds=cfg.eval_thresholds)
    for images, masks, classes in loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images)).cpu()
        acc.update(probs, masks, list(classes))
    return acc.compute()


def train_one_run(cfg, arch: str, seed: int, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_ds, val_ds, test_ds = make_datasets(cfg)
    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False)
    test_loader = _loader(test_ds, cfg, shuffle=False)

    model = build_model(arch, cfg.encoder, cfg.encoder_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    criterion = DiceBCELoss(dice_weight=cfg.dice_weight)

    run_name = f"{arch}_{cfg.encoder}_seed{seed}"
    wb = None
    if cfg.use_wandb:
        try:
            import wandb
            wb = wandb.init(project=cfg.wandb_project, name=run_name,
                            config={**cfg.to_dict(), "arch": arch, "seed": seed}, reinit=True)
        except Exception as e:
            print(f"[wandb disabled: {e}]")

    ckpt_dir = Path(cfg.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_iou, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f"{run_name} e{epoch}", leave=False):
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward(); opt.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_ds)

        val_metrics = evaluate(model, val_loader, device, cfg)
        val_iou = val_metrics["IoU"]
        sched.step(val_iou)
        if wb:
            wb.log({"epoch": epoch, "train_loss": train_loss,
                    "val_IoU": val_iou, "val_Dice": val_metrics["Dice"]})
        print(f"{run_name} e{epoch:02d} loss={train_loss:.4f} valIoU={val_iou:.2f}")

        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"early stop at epoch {epoch} (best valIoU={best_iou:.2f})")
                break

    # load best, evaluate ONCE on held-out test
    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_dir / f"{run_name}.pt")
    test_metrics = evaluate(model, test_loader, device, cfg)
    test_metrics["_best_val_IoU"] = best_iou
    if wb:
        wb.log({f"test_{k}": v for k, v in test_metrics.items() if not isinstance(v, dict)})
        wb.finish()

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8")
    return test_metrics


def run_comparison(cfg, archs=("unet", "deeplabv3plus")):
    results = {}
    for arch in archs:
        per_seed = [train_one_run(cfg, arch, s) for s in cfg.seeds]
        agg = {}
        for k in ("IoU", "Dice", "AUROC", "F1max", "IoU_macro"):
            vals = [m[k] for m in per_seed]
            agg[k] = (round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2))
        results[arch] = {"agg": agg, "per_seed": per_seed}

    # head-to-head table vs paper zero-shot
    pz = cfg.paper_zeroshot
    print("\n================ SEGMENTATION: supervised (ours) vs paper zero-shot ================")
    print(f"{'metric':10s} | {'paper 0-shot':>12s} | "
          + " | ".join(f"{a:>18s}" for a in archs))
    for k in ("IoU", "F1max", "AUROC"):
        row = f"{k:10s} | {pz.get(k, float('nan')):>12.2f} | "
        row += " | ".join(f"{results[a]['agg'][k][0]:>8.2f} +/- {results[a]['agg'][k][1]:<5.2f}"
                          for a in archs)
        print(row)
    print("(framing: FIRST supervised baseline vs paper's zero-shot ceiling — not 'beat their model')")

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "comparison_summary.json").write_text(
        json.dumps({a: results[a]["agg"] for a in archs}, indent=2), encoding="utf-8")
    return results
