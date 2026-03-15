# Concrete Crack Classification

A computer vision project for **binary concrete crack detection** using **PyTorch Lightning**, **MLflow**, and modern image classification models.

This repository compares **ResNet-50** and **ViT-B/16** on crack vs non-crack classification, with support for training, evaluation, and explainability. It is being structured as a **production-oriented ML engineering project**, not just a modeling experiment.

## Features

- Train image classifiers with PyTorch Lightning
- Compare **ResNet-50** and **ViT-B/16**
- Track experiments with **MLflow**
- Evaluate on held-out data with standard classification metrics
- Generate explainability outputs with **Grad-CAM** and **SHAP**
- Run locally or on a GPU cloud environment such as **Runpod**

---

## Repository structure

```text
computer_vision_project/
  configs/
  data/
    raw/
    interim/
    processed/
  docs/
  reports/
  runs/
  src/
    data/
    evaluation/
    explainability/
    models/
    training/
  tests/
  mlruns/
  README.md
  requirements.txt
```

---

## Models

The project currently supports:

- **ResNet-50**
- **ViT-B/16**

Model implementations are under:

- `src/models/`

Training entrypoint:

- `src/training/train.py`

Evaluation entrypoint:

- `src/evaluation/evaluate.py`

Explainability entrypoint:

- `src/explainability/run_explainability.py`

---

## Data

The repository expects image datasets under:

```text
data/raw/
```

Typical layout:

```text
data/raw/
  sdnet2018/
    ...
  ccic/
    ...
```

### Important

Raw datasets are **not included** in this repository.

You must download or copy them separately before training.

### Raw data guidelines

- keep raw data out of Git
- do not mix reports, checkpoints, or predictions into `data/raw/`
- keep original source images unchanged when possible

---

## Setup

### 1. Clone the repository

````bash
git clone <your-repository-url>
cd computer_vision_project