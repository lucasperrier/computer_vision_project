from __future__ import annotations

from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

from src.config.load import to_runtime_config
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


def build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    name = str(model_cfg.get("name", "resnet50")).lower()
    if "vit" in name:
        return VisionTransformerModule(model_cfg)
    return ResNet50Module(model_cfg)


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    runtime = to_runtime_config(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if not runtime.checkpoint_path:
        raise ValueError("checkpoint_path is required for inference")

    model_cls = VisionTransformerModule if "vit" in runtime.model.name else ResNet50Module
    model = model_cls.load_from_checkpoint(runtime.checkpoint_path, config=cfg_dict["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and runtime.device != "cpu" else "cpu")
    model.to(device)

    # Example single-image path; override with CLI:
    # python -m src.inference.predict image_path=/abs/path/image.jpg checkpoint_path=/abs/path.ckpt
    image_path = cfg.get("image_path", None)
    if image_path is None:
        raise ValueError("Please pass image_path=... as Hydra override")

    tfm = make_transform(runtime.data.image_size)
    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    print({
        "image_path": image_path,
        "predicted_class": pred,
        "probabilities": probs.detach().cpu().tolist(),
        "model": runtime.model.name,
        "checkpoint_path": runtime.checkpoint_path,
    })


if __name__ == "__main__":
    main()