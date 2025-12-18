from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SHAPOutput:
    """
    values: SHAP attributions.
      - superpixel SHAP: typically (B, H, W, 3) or (B, H, W, C)
      - patch SHAP: (B, H_patches, W_patches)
    base_values: expected value(s) from SHAP
    target_class: explained classes
    """
    values: np.ndarray
    base_values: np.ndarray
    target_class: Union[int, np.ndarray]


def make_predict_fn_from_torch(
    model: torch.nn.Module,
    device: torch.device,
    *,
    assumes_uint8_rgb: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns predict(images_np) -> probs_np for SHAP.

    SHAP image maskers typically feed uint8 RGB images (B,H,W,3).
    Your training expects normalized tensors:
      mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] and NCHW.

    This function converts uint8->float, normalizes, and runs model.
    """
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def predict(images_np: np.ndarray) -> np.ndarray:
        if images_np.ndim == 3:
            images_np_b = images_np[None, ...]
        else:
            images_np_b = images_np

        x = torch.from_numpy(images_np_b).to(device)

        # (B,H,W,3) -> (B,3,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()

        if assumes_uint8_rgb:
            x = x.float() / 255.0
        else:
            x = x.float()

        x = (x - mean) / std

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    return predict


def shap_explain_superpixels(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    images_uint8: np.ndarray,
    *,
    target_class: Optional[int] = None,
    max_evals: int = 500,
    batch_size: int = 16,
    segmentation: str = "slic",
) -> SHAPOutput:
    """
    Superpixel SHAP using shap.maskers.Image.
    Realistic usage: small number of images (e.g., 1..20) because it's slow.
    """
    import shap

    if images_uint8.ndim == 3:
        images_uint8 = images_uint8[None, ...]

    masker = shap.maskers.Image(segmentation, images_uint8[0].shape)
    explainer = shap.Explainer(predict_fn, masker)

    if target_class is None:
        probs = predict_fn(images_uint8)
        target_arr = probs.argmax(axis=1)
    else:
        target_arr = np.full((images_uint8.shape[0],), int(target_class), dtype=np.int64)

    shap_values = explainer(
        images_uint8,
        max_evals=max_evals,
        batch_size=batch_size,
        outputs=target_arr.tolist(),
    )

    return SHAPOutput(
        values=np.array(shap_values.values),
        base_values=np.array(shap_values.base_values),
        target_class=target_arr,
    )


def shap_explain_vit_patches_kernel(
    model: torch.nn.Module,
    device: torch.device,
    image_t: torch.Tensor,
    *,
    patch_size: int = 16,
    vit_grid: Tuple[int, int] = (14, 14),
    target_class: Optional[int] = None,
    nsamples: int = 300,
    baseline: str = "blur",
) -> SHAPOutput:
    """
    Patch-level KernelSHAP for a *single* image.

    Why single image? Kernel SHAP cost grows with #features (196 patches) and nsamples.
    This is realistic for a project demo but not for large-scale evaluation.

    image_t: (3,224,224) or (1,3,224,224) normalized like your datamodule output.
    Returns: values shaped (1, H_p, W_p).
    """
    import shap

    model.eval()

    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0)

    image_t = image_t.to(device)
    B, C, H, W = image_t.shape
    H_p, W_p = vit_grid
    M = H_p * W_p

    if H_p * patch_size != H or W_p * patch_size != W:
        raise ValueError("vit_grid and patch_size must match the image resolution (e.g., 224=14*16).")

    # baseline image in model space (already normalized!)
    with torch.no_grad():
        if baseline == "zero":
            base = torch.zeros_like(image_t)
        else:
            # cheap blur in normalized space: average over patches and upsample
            pooled = F.avg_pool2d(image_t, kernel_size=patch_size, stride=patch_size)
            base = F.interpolate(pooled, size=(H, W), mode="nearest")

    with torch.no_grad():
        logits = model(image_t)
        if target_class is None:
            tgt = int(logits.argmax(dim=1).item())
        else:
            tgt = int(target_class)

    def apply_mask(mask01: np.ndarray) -> torch.Tensor:
        # mask01: (M,) where 1 keeps original patch, 0 uses baseline patch
        m = torch.from_numpy(mask01.reshape(H_p, W_p)).float().to(device)
        m_up = m.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)  # (H,W)
        m_up = m_up.view(1, 1, H, W)
        return image_t * m_up + base * (1.0 - m_up)

    def predict_from_masks(masks: np.ndarray) -> np.ndarray:
        # masks: (K, M)
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for mask in masks:
                x_masked = apply_mask(mask.astype(np.float32))
                probs = F.softmax(model(x_masked), dim=1)
                outs.append(probs.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    # background masks: a few random masks for KernelExplainer baseline
    bg = np.random.binomial(1, 0.5, size=(min(20, M), M)).astype(np.float32)

    ke = shap.KernelExplainer(predict_from_masks, bg)

    # explain the "all-ones" mask (original image) in mask space
    shap_vals = ke.shap_values(np.ones((1, M), dtype=np.float32), nsamples=nsamples)

    if isinstance(shap_vals, list):
        vals_m = shap_vals[tgt][0]
        base_val = ke.expected_value[tgt]
    else:
        vals_m = shap_vals[0]
        base_val = ke.expected_value

    return SHAPOutput(
        values=vals_m.reshape(1, H_p, W_p),
        base_values=np.array([base_val]),
        target_class=tgt,
    )