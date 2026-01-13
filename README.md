# Computer Vision Project — Crack Detection (ResNet-50 vs ViT)

This repository contains the code and artifacts for a concrete crack detection project comparing a CNN baseline (ResNet-50) and a Vision Transformer (ViT-B/16) under a unified pipeline.

It includes:

- Training with PyTorch Lightning + Hydra configs
- Experiment tracking with MLflow (SQLite backend)
- Evaluation artifacts (metrics JSON, confusion matrices, predictions)
- Explainability (Grad-CAM + SHAP) for both ResNet and ViT

## Results (test split)

Metrics below come from the generated `reports/**/metrics.json` files.

| Model / Condition | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| ViT (fine-tuned) | **0.9978** | **0.9978** | **0.99997** |
| ResNet (fine-tuned) | 0.8572 | 0.8486 | 0.9483 |
| ViT (pretrained only, no fine-tuning) | 0.3115 | 0.4087 | 0.2515 |
| ResNet (pretrained only, no fine-tuning) | 0.2040 | 0.2386 | 0.1436 |

## Repository layout

- `configs/`: YAML experiment configs (train/eval/explainability)
- `src/`:
	- `data/`: Lightning `CrackDataModule`
	- `models/`: `ResNet50Module`, `VisionTransformerModule`
	- `training/`: training entrypoint
	- `evaluation/`: evaluation scripts (metrics + artifacts)
	- `explainability/`: Grad-CAM + SHAP utilities and runner
- `reports/`: generated outputs (metrics, confusion matrices, predictions, explanations)
- `mlruns/`: MLflow tracking directory
- `mlflow.db`: MLflow SQLite database
- `tests/`: unit tests

## Environment setup

```bash
pip install -r requirements.txt
```

## Data

Place datasets under `data/raw/`.

The project was run with SDNET2018-style and CCIC-style folder structures. At a minimum, ensure you have positive/negative (or crack/no-crack) subfolders.

## Training

Training is driven by YAML configs in `configs/`.

```bash
python -m src.training.train --config configs/train_vit.yaml
python -m src.training.train --config configs/train_resnet.yaml
```

Outputs:

- MLflow runs in `mlruns/` and `mlflow.db`
- checkpoints/artifacts under `runs/` and MLflow artifacts

## Evaluation

Evaluation writes artifacts under `reports/` (including `metrics.json`, confusion matrix, and saved predictions).

```bash
python -m src.evaluation.evaluate --config configs/eval_vit.yaml
python -m src.evaluation.evaluate --config configs/eval_resnet.yaml
```

Pretrained-only baselines (no fine-tuning):

```bash
python -m src.evaluation.evaluate --config configs/eval_baseline_vit.yaml
python -m src.evaluation.evaluate --config configs/eval_baseline_resnet.yaml
```

## Explainability

Explainability outputs (overlays and heatmaps) are written under `reports/**/explanations/`.

```bash
python -m src.explainability.run_explainability --config configs/run_explainability_vit.yaml
python -m src.explainability.run_explainability --config configs/run_explainability_resnet.yaml
```

## Report

The final report is written in LaTeX and references figures under `report/figures/`.

Evaluation metrics used in the report are stored in:

- `reports/eval_trained_vit/metrics.json`
- `reports/eval_trained_resnet/metrics.json`
- `reports/eval_pretrained_vit/metrics.json`
- `reports/eval_pretrained_resnet/metrics.json`

## References

- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. https://arxiv.org/abs/2010.11929
- Golding, V.P.; Gharineiat, Z.; Munawar, H.S.; Ullah, F. *Crack Detection in Concrete Structures Using Deep Learning*. Sustainability 2022, 14, 8117. https://doi.org/10.3390/su14138117
- Kashefi, R.; Barekatain, L.; Sabokrou, M.; Aghaeipoor, F. *Explainability of Vision Transformers: A Comprehensive Review and New Perspectives*. arXiv 2023. https://arxiv.org/abs/2311.06786
- Selvaraju, R.R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. IJCV 2020, 128, 336--359. https://doi.org/10.1007/s11263-019-01228-7
- Zhang, X. et al. *Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets*. arXiv 2025. https://arxiv.org/abs/2508.10256