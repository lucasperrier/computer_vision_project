from __future__ import annotations
from typing import Any, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_pre_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pre = cfg.get("preprocessing", {})
    return {
        "image_size": int(pre.get("image_size", 224)),
        "mean": pre.get("mean", [0.485, 0.456, 0.406]),
        "std": pre.get("std", [0.229, 0.224, 0.225]),
        "hflip_p": float(pre.get("hflip_p", 0.5)),
        "brightness_contrast_p": float(pre.get("brightness_contrast_p", 0.3)),
        "shift_scale_rotate_p": float(pre.get("shift_scale_rotate_p", 0.3)),
        "shift_limit": float(pre.get("shift_limit", 0.03)),
        "scale_limit": float(pre.get("scale_limit", 0.05)),
        "rotate_limit": int(pre.get("rotate_limit", 10)),
    }


def build_train_transforms(cfg: Dict[str, Any]) -> A.Compose:
    p = _get_pre_cfg(cfg)
    return A.Compose([
        A.Resize(p["image_size"], p["image_size"]),
        A.HorizontalFlip(p=p["hflip_p"]),
        A.RandomBrightnessContrast(p=p["brightness_contrast_p"]),
        A.ShiftScaleRotate(
            shift_limit=p["shift_limit"],
            scale_limit=p["scale_limit"],
            rotate_limit=p["rotate_limit"],
            p=p["shift_scale_rotate_p"],
        ),
        A.Normalize(mean=p["mean"], std=p["std"]),
        ToTensorV2(),
    ])


def build_val_transforms(cfg: Dict[str, Any]) -> A.Compose:
    p = _get_pre_cfg(cfg)
    return A.Compose([
        A.Resize(p["image_size"], p["image_size"]),
        A.Normalize(mean=p["mean"], std=p["std"]),
        ToTensorV2(),
    ])


def build_eval_transforms(cfg: Dict[str, Any]) -> A.Compose:
    return build_val_transforms(cfg)


def build_inference_transforms(cfg: Dict[str, Any]) -> A.Compose:
    return build_val_transforms(cfg)