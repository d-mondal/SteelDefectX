"""
Build the segmentation model via segmentation-models-pytorch (smp).

Both architectures draw the SAME ImageNet-pretrained encoder, so a U-Net vs
DeepLabV3+ comparison isolates the DECODER as the only difference — a genuinely
fair head-to-head. This is fine-tuning a pretrained encoder + training a fresh
decoder; nothing is pretrained from scratch (a locked project decision).

`dropout` enables decoder dropout, which is what makes MC-Dropout uncertainty
possible: at inference we keep dropout ACTIVE and run several stochastic passes.
"""
from __future__ import annotations
import torch.nn as nn
import segmentation_models_pytorch as smp

_ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(arch: str, encoder: str = "resnet34", encoder_weights: str = "imagenet",
                dropout: float = 0.0):
    arch = arch.lower()
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch '{arch}', choose from {list(_ARCHS)}")
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,      # gray replicated to 3ch upstream
        classes=1,          # binary defect mask; logits out, apply sigmoid at eval
        activation=None,
    )
    # smp exposes decoder dropout via aux/decoder params on U-Net; pass it when >0.
    if dropout and arch == "unet":
        # smp>=0.3 U-Net accepts `decoder_use_batchnorm` etc; dropout is added via the
        # segmentation head. We attach a dropout layer to the head for MC-Dropout.
        model = _ARCHS[arch](**kwargs)
        _inject_head_dropout(model, p=dropout)
        return model
    return _ARCHS[arch](**kwargs)


def _inject_head_dropout(model, p: float):
    """Prepend a Dropout2d before the segmentation head so MC-Dropout has a stochastic
    layer that stays active at inference. Kept simple + explicit so it's easy to defend."""
    head = model.segmentation_head
    model.segmentation_head = nn.Sequential(nn.Dropout2d(p=p), *list(head))


def enable_mc_dropout(model):
    """Put the model in eval mode but re-activate only Dropout layers — the core trick
    of MC-Dropout: batchnorm stays in eval (stable stats) while dropout stays stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model