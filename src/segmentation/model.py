"""
Build the segmentation model via segmentation-models-pytorch (smp).

Both architectures draw the SAME ImageNet-pretrained encoder, so a U-Net vs
DeepLabV3+ comparison isolates the DECODER as the only difference — a genuinely
fair head-to-head. This is fine-tuning a pretrained encoder + training a fresh
decoder; nothing is pretrained from scratch (a locked project decision).
"""
from __future__ import annotations
import segmentation_models_pytorch as smp

_ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(arch: str, encoder: str = "resnet34", encoder_weights: str = "imagenet"):
    arch = arch.lower()
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {list(_ARCHS)}")
    return _ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,      # gray replicated to 3ch upstream
        classes=1,          # binary defect mask; logits out, apply sigmoid at eval
        activation=None,
    )
