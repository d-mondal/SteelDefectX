"""Single source of truth for segmentation experiment settings."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    # --- data ---
    data_root: str = "data"                 # set to the HF snapshot dir on Colab
    train_img_dir: str = "train"            # relative to data_root
    train_mask_dir: str = "train_mask"
    test_img_dir: str = "val"               # official val == our held-out TEST
    test_mask_dir: str = "val_mask"
    split_dir: str = "data/splits"          # from make_split.py
    image_size: int = 256

    # --- model (same encoder for BOTH archs -> fair head-to-head) ---
    arch: str = "unet"                      # "unet" | "deeplabv3plus"
    encoder: str = "resnet34"               # ImageNet-pretrained; supported by both
    encoder_weights: str = "imagenet"

    # --- optimisation ---
    batch_size: int = 16
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dice_weight: float = 0.5                # loss = dice_weight*Dice + (1-dice_weight)*BCE
    early_stop_patience: int = 8            # epochs w/o internal-val IoU improvement
    num_workers: int = 2

    # --- reproducibility / comparison ---
    seeds: tuple = (0, 1, 2)                # 3 seeds -> mean +/- std on the NEW result
    eval_thresholds: int = 50               # for AUROC / F1-max threshold sweep

    # --- tracking / io ---
    use_wandb: bool = False                 # opt-in; never blocks a run
    wandb_project: str = "steeldefectx-seg"
    out_dir: str = "results/segmentation"
    ckpt_dir: str = "models/segmentation"

    # paper's best ZERO-SHOT segmentation numbers (Table 3) — the target to beat,
    # framed as "first supervised baseline vs paper zero-shot", never "beat their model"
    paper_zeroshot: dict = field(default_factory=lambda: {
        "AUROC": 88.21, "F1max": 53.36, "IoU": 37.49,
    })

    def resolved(self, key: str) -> Path:
        return Path(self.data_root) / getattr(self, key)

    def to_dict(self):
        return asdict(self)
