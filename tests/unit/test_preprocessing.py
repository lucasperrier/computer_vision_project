import numpy as np
import torch

from src.preprocessing.transforms import (
    build_train_transforms,
    build_val_transforms,
    build_eval_transforms,
    build_inference_transforms,
)


def _dummy_image(h=300, w=400, c=3):
    return np.random.randint(0, 256, size=(h, w, c), dtype=np.uint8)


def test_train_transforms_output_tensor_shape_and_dtype():
    cfg = {"preprocessing": {"image_size": 224}}
    t = build_train_transforms(cfg)

    out = t(image=_dummy_image())["image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_val_eval_inference_have_same_deterministic_output():
    cfg = {
        "preprocessing": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    }
    img = _dummy_image()

    val_t = build_val_transforms(cfg)
    eval_t = build_eval_transforms(cfg)
    inf_t = build_inference_transforms(cfg)

    val_out = val_t(image=img)["image"]
    eval_out = eval_t(image=img)["image"]
    inf_out = inf_t(image=img)["image"]

    assert torch.allclose(val_out, eval_out)
    assert torch.allclose(val_out, inf_out)
    assert val_out.shape == (3, 224, 224)


def test_defaults_work_without_preprocessing_block():
    cfg = {}
    t = build_val_transforms(cfg)
    out = t(image=_dummy_image())["image"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)