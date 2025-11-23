# Explainable Crack Detection with Vision Transformers and CNNs

## Project Overview
This project benchmarks a Vision Transformer (ViT) against a CNN (ResNet-50) for crack detection in concrete, integrating explainability tools (Grad-CAM, SHAP) and a dashboard for engineering decision support.

## Execution roadmap

### Environment & data intake
- Pin dependencies (PyTorch, Lightning, timm, torchvision, Captum, SHAP, albumentations, Streamlit, wandb).
- Download SDNET2018 + CCIC into data/raw; script checksums and metadata catalog (src/data/download.py).
- Build preprocessing notebook to inspect label balance, lighting conditions, and standardize resolution (e.g., 224×224).

### Data module & augmentation
- Implement CrackDataModule handling combined dataset splits (train/val/test + robustness hold-out).
- Add augmentations (random rotations, color jitter for lighting, Gaussian noise) via albumentations; ensure deterministic val/test transforms.
- Unit-test dataloader shapes, normalization stats, and class distribution.

### Baseline CNN training
- Define ResNet50Module using LightningModule with configurable optimizer/scheduler.
- Train with early stopping + checkpoints; log metrics (accuracy/F1/AUC) and confusion matrix per epoch.
- Save best weights and sample Grad-CAM maps to confirm localization quality.

### ViT fine-tuning
- Load ImageNet-pretrained ViT (e.g., vit_base_patch16_224) with head adapted for binary classification.
- Experiment with layer freezing/unfreezing schedule; use mixup/cutmix if beneficial.
- Track same metrics plus attention rollout visualizations; ensure >95% accuracy target feasibility.

### Explainability suite
- Build reusable Grad-CAM wrapper (supports CNN & ViT) and SHAP pipeline (sampling background sets).
- Quantify localization accuracy (IoU or pointing game) using available crack masks or proxy bounding boxes if dataset lacks them (can annotate small subset).
- Implement faithfulness tests (remove top-k pixels vs random).

### Robustness evaluations
- Craft synthetic lighting/material perturbations (brightness shifts, texture overlays) and measure metric deltas for both models.
- Summarize robustness gap tables + statistical tests if possible.

### Dashboard & reporting
- Streamlit app: model selector, upload/custom image, show prediction, Grad-CAM heatmap, SHAP explanation, and metric summaries.
- Generate final report notebook exporting plots (ROC curves, attention maps, robustness charts) into reports/.

### Automation & CI
- Add CLI entrypoints (python -m src.training.train --config configs/model_vit.yaml, etc.).
- Write unit/integration tests for data splits, metrics, explainability outputs (e.g., heatmap shape, saliency sum).
- Optional: GitHub Actions to lint (ruff), run smoke tests on CPU subset.

## Work Split (Two Contributors)

**Contributor 1:**
- Data intake, preprocessing, augmentation
- Baseline CNN (ResNet-50) implementation and training
- Grad-CAM explainability for CNN
- Initial dashboard setup

**Contributor 2:**
- ViT fine-tuning and training
- SHAP explainability for ViT and CNN
- Robustness evaluation
- Dashboard completion and reporting

Both contributors should collaborate on data module, evaluation metrics, and final report.

## Directory Structure
```
configs/
data/
notebooks/
src/
dashboard/
tests/
reports/
docs/
```

## Getting Started
```bash
pip install -r requirements.txt
```

## References
- SDNET2018, CCIC datasets
- Dosovitskiy et al., "An Image is Worth 16x16 Words"
- Selvaraju et al., "Grad-CAM"
- Kashefi et al., "Explainability of Vision Transformers"
``