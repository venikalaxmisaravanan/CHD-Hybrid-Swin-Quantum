"""Grad-CAM for the hybrid model, with a quantitative anatomical check.

Two things here that matter for the paper:

1. Swin needs a reshape_transform. pytorch-grad-cam expects (B, C, H, W)
   activations; Swin blocks emit either (B, L, C) or (B, H, W, C) depending on
   the timm version. Without this the CAM is silently wrong, not an error.

2. The overlap score turns interpretability from a figure into a number.
   HVSMR-2.0 ships substructure segmentation masks, so you can measure what
   fraction of CAM mass falls inside annotated cardiac anatomy instead of
   asserting it from visual inspection. Reviewers weight a measured claim far
   more heavily, and this costs you nothing because the labels are already there.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise Swin block output to (B, C, H, W) for pytorch-grad-cam."""
    if tensor.dim() == 4:                      # (B, H, W, C)
        return tensor.permute(0, 3, 1, 2).contiguous()
    if tensor.dim() == 3:                      # (B, L, C)
        b, l, c = tensor.shape
        h = int(round(l ** 0.5))
        while h > 1 and l % h != 0:
            h -= 1
        w = l // h
        return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unexpected activation shape {tuple(tensor.shape)}")


def target_layer(model):
    """Final Swin block's first norm. Verify against your timm version:
    print(model.backbone) and confirm the attribute path before trusting output."""
    return model.backbone.layers[-1].blocks[-1].norm1


def generate_cam(model, x: torch.Tensor, class_idx: int, device="cuda"):
    """Return a (H, W) CAM in [0,1] for the requested severity class."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    cam = GradCAM(model=model,
                  target_layers=[target_layer(model)],
                  reshape_transform=swin_reshape_transform)
    out = cam(input_tensor=x.to(device),
              targets=[ClassifierOutputTarget(class_idx)])
    return out[0]


def cam_mask_overlap(cam: np.ndarray, mask_slice: np.ndarray,
                     top_frac: float = 0.2) -> dict:
    """Fraction of the strongest CAM region that lands on annotated anatomy.

    cam         (H, W) in [0, 1]
    mask_slice  (H', W') segmentation labels, 0 = background
    top_frac    proportion of pixels treated as the CAM's active region
    """
    m = torch.from_numpy((mask_slice > 0).astype(np.float32))[None, None]
    m = F.interpolate(m, size=cam.shape, mode="nearest")[0, 0].numpy() > 0.5

    k = max(1, int(cam.size * top_frac))
    thresh = np.partition(cam.ravel(), -k)[-k]
    active = cam >= thresh

    inter = float((active & m).sum())
    return {
        "precision": inter / max(active.sum(), 1),   # CAM mass on anatomy
        "recall": inter / max(m.sum(), 1),           # anatomy covered by CAM
        "iou": inter / max((active | m).sum(), 1),
        "mask_frac": float(m.mean()),                # chance-level baseline
    }


def summarise_overlap(records: list[dict]) -> dict:
    """Aggregate per-slice overlap. Compare mean precision against mask_frac:
    precision at or below mask_frac means the CAM is no better than chance."""
    if not records:
        return {}
    keys = records[0].keys()
    return {f"mean_{k}": float(np.mean([r[k] for r in records])) for k in keys}
