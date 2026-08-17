"""
Dataset for supervised BINARY defect segmentation on SteelDefectX.

Key design points (locked decisions):
  - Masks are BINARY (defect vs background), thresholded at >127. One shared model
    across all 25 classes; we do NOT do 25-way pixel semantic segmentation.
  - We still carry each image's CLASS LABEL through the dataset so evaluation can
    report per-class IoU/Dice by grouping predictions afterwards (retains all 25
    classes, including the rare ones).
  - Steel images are grayscale but the ImageNet-pretrained encoder expects 3 channels,
    so the single gray channel is replicated to RGB before ImageNet normalisation.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as e:  # pragma: no cover
    A = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    """Light, defect-safe augmentation. Flips/rotations are fine for surface texture;
    we avoid aggressive colour jitter that could destroy subtle low-contrast defects."""
    if A is None:
        raise ImportError("albumentations is required: pip install albumentations")
    common_tail = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            *common_tail,
        ])
    return A.Compose([A.Resize(image_size, image_size), *common_tail])


def _load_ids(path: Path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _class_map_from_text(text_json: Path):
    """image_name -> class_name, from the dataset's *-text.json."""
    with open(text_json, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else list(data.values())
    e0 = entries[0]
    name_key = next((k for k in e0 if "image" in k.lower() and "name" in k.lower()), "image_name")
    class_key = next((k for k in e0 if k.lower() in ("class_name", "class", "classname", "label")), "class_name")
    return {e[name_key]: e[class_key] for e in entries}


class SteelDefectSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, class_map, transform):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = list(image_names)
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def _mask_path(self, stem):
        # masks are PNG (confirmed in audit); fall back to jpg just in case
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.mask_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"no mask for {stem} in {self.mask_dir}")

    def __getitem__(self, i):
        name = self.names[i]
        stem = Path(name).stem
        img = cv2.imread(str(self.image_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.image_dir / name)
        img = np.stack([img, img, img], axis=-1)  # gray -> 3ch for ImageNet encoder

        mask = cv2.imread(str(self._mask_path(stem)), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)     # binarise

        out = self.transform(image=img, mask=mask)
        image = out["image"]                        # (3,H,W) float
        mask = out["mask"].unsqueeze(0).float()     # (1,H,W) in {0,1}
        cls = self.class_map.get(name, "unknown")
        return image, mask, cls


def make_datasets(cfg):
    """Build train / internal-val / test datasets from cfg + frozen split files."""
    split_dir = Path(cfg.split_dir)
    train_ids = _load_ids(split_dir / "train_ids.txt")
    val_ids = _load_ids(split_dir / "internalval_ids.txt")

    train_text = cfg.resolved("train_img_dir").parent / "train-text.json"
    test_text = cfg.resolved("test_img_dir").parent / "val-text.json"
    train_cmap = _class_map_from_text(train_text)
    test_cmap = _class_map_from_text(test_text)

    ti, tm = cfg.resolved("train_img_dir"), cfg.resolved("train_mask_dir")
    xi, xm = cfg.resolved("test_img_dir"), cfg.resolved("test_mask_dir")

    tf_train = build_transforms(cfg.image_size, train=True)
    tf_eval = build_transforms(cfg.image_size, train=False)

    train_ds = SteelDefectSegDataset(ti, tm, train_ids, train_cmap, tf_train)
    val_ds = SteelDefectSegDataset(ti, tm, val_ids, train_cmap, tf_eval)
    test_ds = SteelDefectSegDataset(xi, xm, list(test_cmap.keys()), test_cmap, tf_eval)
    return train_ds, val_ds, test_ds
