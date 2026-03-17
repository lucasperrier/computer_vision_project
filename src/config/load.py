from __future__ import annotations
from omegaconf import DictConfig, OmegaConf
from src.config.schema import RuntimeConfig


def to_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """
    Convert Hydra/OmegaConf config into validated typed RuntimeConfig.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    return RuntimeConfig.model_validate(raw)