import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import yaml

from src.data.datamodule import CrackDataModule
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule

from src.explainability.grad_cam import GradCAM, upsample_cam_to_image
from src.explainability.shap_utils import (
    make_predict_fn_from_torch,
    shap_explain_superpixels,
    shap_explain_vit_patches_kernel,
)
from src.explainability.viz import (
    denormalize_imagenet,
    to_uint8_hwc,
    colorize_heatmap,
    overlay_heatmap_on_image,
    save_png,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]):
    model_name = str(cfg.get("model", "resnet50")).lower()
    if "vit" in model_name:
        return VisionTransformerModule(cfg)
    return ResNet50Module(cfg)


def load_from_checkpoint_if_any(cfg: Dict[str, Any], model):
    ckpt_path = cfg.get("checkpoint_path", None)
    if not ckpt_path:
        return model

    ckpt_path = str(ckpt_path)
    model_name = str(cfg.get("model", "")).lower()
    model_cls = VisionTransformerModule if "vit" in model_name else ResNet50Module
    return model_cls.load_from_checkpoint(ckpt_path, config=cfg)


def pick_gradcam_target_layer(model_name: str, timm_model: torch.nn.Module) -> torch.nn.Module:
    """
    Select a reasonable default target layer for timm models.
    """
    model_name = model_name.lower()

    if "vit" in model_name:
        # token stream hook; outputs (B,N,D)
        return timm_model.blocks[-1].norm1

    # timm resnet usually exposes .layer4
    if hasattr(timm_model, "layer4"):
        return timm_model.layer4[-1]

    # fallback: last child module
    children = list(timm_model.children())
    if not children:
        raise ValueError("Cannot find a target layer automatically.")
    return children[-1]


def vit_grid_from_name(model_name: str) -> Tuple[int, int]:
    """
    For vit_base_patch16_224 => 224/16 = 14.
    If you change model/resolution, update this.
    """
    # minimal, aligned with your configs
    return (14, 14)


def main(config_path: str):
    cfg = load_config(config_path)

    output_dir = Path(cfg.get("output_dir", "reports/explainability"))
    out_dir = output_dir / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    datamodule = CrackDataModule(
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        val_split=float(cfg.get("val_split", 0.1)),
        test_split=float(cfg.get("test_split", 0.1)),
        robustness_split=float(cfg.get("robustness_split", 0.1)),
    )
    datamodule.setup(stage="test")
    dl = datamodule.test_dataloader()

    # Model
    model = build_model(cfg)
    model = load_from_checkpoint_if_any(cfg, model)
    model.eval().to(device)

    timm_model = model.model  # your LightningModules store timm model here
    model_name = str(cfg.get("model", "resnet50"))

    # Explainability settings
    n_images = int(cfg.get("n_explain_images", 12))
    alpha = float(cfg.get("overlay_alpha", 0.45))

    # Grad-CAM setup
    target_layer = pick_gradcam_target_layer(model_name, timm_model)
    gc = GradCAM(timm_model, target_layer=target_layer)

    # SHAP setup
    # For superpixel SHAP we need uint8 RGB; predict_fn handles normalization internally
    predict_fn = make_predict_fn_from_torch(timm_model, device)

    do_shap = bool(cfg.get("do_shap", True))
    do_shap_vit_patch = bool(cfg.get("do_shap_vit_patch", False))
    shap_max_evals = int(cfg.get("shap_max_evals", 500))
    shap_batch_size = int(cfg.get("shap_batch_size", 16))

    # Iterate over test data, save explanations
    saved = 0
    for batch_idx, batch in enumerate(dl):
        x, y = batch
        x = x.to(device)

        B = x.shape[0]
        for i in range(B):
            if saved >= n_images:
                break

            xi = x[i : i + 1]  # (1,3,224,224)
            yi = int(y[i])

            # Convert to displayable RGB
            img01 = denormalize_imagenet(xi[0]).detach()
            img_u8 = to_uint8_hwc(img01)  # (224,224,3)

            # ---------- Grad-CAM ----------
            if "vit" in model_name.lower():
                cam_out = gc(xi, vit_grid=vit_grid_from_name(model_name))
            else:
                cam_out = gc(xi)

            cam224 = upsample_cam_to_image(cam_out.cam, (224, 224))[0, 0].detach().cpu().numpy()  # (224,224)
            hm_u8 = colorize_heatmap(cam224)
            overlay = overlay_heatmap_on_image(img_u8, hm_u8, alpha=alpha)

            stem = f"sample_{saved:04d}_label{yi}_pred{int(cam_out.class_idx[0].item())}"
            save_png(out_dir / f"{stem}_image.png", img_u8)
            save_png(out_dir / f"{stem}_gradcam_overlay.png", overlay)
            save_png(out_dir / f"{stem}_gradcam_heatmap.png", hm_u8)

            # ---------- SHAP (superpixels) ----------
            if do_shap:
                # shap expects (B,H,W,3) uint8
                shap_out = shap_explain_superpixels(
                    predict_fn,
                    img_u8[None, ...],
                    target_class=None,
                    max_evals=shap_max_evals,
                    batch_size=shap_batch_size,
                    segmentation=str(cfg.get("shap_segmentation", "slic")),
                )
                # shap_out.values typically (1,H,W,3). Convert to intensity for visualization.
                vals = shap_out.values
                if vals.ndim == 4:
                    # aggregate channels
                    shap_map = np.mean(np.abs(vals[0]), axis=2)  # (H,W)
                else:
                    shap_map = np.abs(vals[0])

                # normalize to [0,1]
                shap_map = shap_map - shap_map.min()
                shap_map = shap_map / (shap_map.max() + 1e-6)

                shap_hm_u8 = colorize_heatmap(shap_map)
                shap_overlay = overlay_heatmap_on_image(img_u8, shap_hm_u8, alpha=alpha)
                save_png(out_dir / f"{stem}_shap_superpix_overlay.png", shap_overlay)
                save_png(out_dir / f"{stem}_shap_superpix_heatmap.png", shap_hm_u8)

            # ---------- SHAP (ViT patch KernelSHAP) ----------
            # WARNING: slow. Default is off.
            if do_shap_vit_patch and "vit" in model_name.lower():
                patch_out = shap_explain_vit_patches_kernel(
                    timm_model,
                    device,
                    xi[0].detach(),  # (3,224,224)
                    patch_size=int(cfg.get("vit_patch_size", 16)),
                    vit_grid=vit_grid_from_name(model_name),
                    nsamples=int(cfg.get("vit_patch_shap_nsamples", 300)),
                    baseline=str(cfg.get("vit_patch_shap_baseline", "blur")),
                )
                patch_map = np.abs(patch_out.values[0])  # (14,14)
                patch_map = patch_map - patch_map.min()
                patch_map = patch_map / (patch_map.max() + 1e-6)

                # upsample to 224x224 for overlay
                patch_map_t = torch.from_numpy(patch_map).float()[None, None, :, :]
                patch_map_224 = torch.nn.functional.interpolate(
                    patch_map_t, size=(224, 224), mode="bilinear", align_corners=False
                )[0, 0].numpy()

                patch_hm_u8 = colorize_heatmap(patch_map_224)
                patch_overlay = overlay_heatmap_on_image(img_u8, patch_hm_u8, alpha=alpha)
                save_png(out_dir / f"{stem}_shap_vitpatch_overlay.png", patch_overlay)
                save_png(out_dir / f"{stem}_shap_vitpatch_heatmap.png", patch_hm_u8)

            saved += 1

        if saved >= n_images:
            break

    gc.close()
    print(f"Saved {saved} explanation sets to: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Generate Grad-CAM and SHAP visual explanations.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (reuse eval config).")
    args = p.parse_args()
    main(args.config)