# SentinelInspect
### Reliability-First Visual Inspection ML System

SentinelInspect is a **data-centric, reliability-first computer vision system** for visual inspection tasks, built around a structural crack classification use case.

The project began as a reproducible crack classification pipeline and is being extended into a broader **ML engineering and applied research system** focused on:

- deterministic and auditable dataset construction
- robust evaluation under noise, artifacts, and domain shift
- reliability analysis beyond accuracy
- data-efficient learning under limited labels
- reusable inference for batch and API workflows
- MLflow-managed lifecycle and release evaluation

The goal is not just to train a classifier, but to build a **credible end-to-end visual inspection ML system** that demonstrates production-oriented engineering and research-grade evaluation.

---

## Why this project

In many real inspection settings, accuracy on a clean held-out split is not enough.

A useful system must also answer questions such as:

- What exactly was the dataset used for training?
- Are the train/validation/test splits deterministic and auditable?
- How sensitive is the model to blur, compression, noise, or brightness changes?
- How does performance change under source shift or out-of-distribution inputs?
- When should the model abstain or be considered unreliable?
- Can performance be maintained with fewer labels?
- Can the model be served, monitored, and promoted through explicit release gates?

SentinelInspect is designed to make those questions part of the core workflow.

---

## Current scope

Current repository focus:

- binary image classification for structural crack detection
- transfer learning with CNN and ViT backbones
- experiment tracking with MLflow
- PyTorch Lightning-based training
- shared preprocessing and typed config loading
- offline evaluation and explainability analysis
- scaffolding for inference, API serving, monitoring, and model lifecycle

Target direction:

- deterministic data contracts
- robustness benchmarks
- calibration and selective prediction
- OOD and slice-based evaluation
- active learning, pseudo-labeling, and distillation
- batch and online inference with monitoring
- release gating and model promotion

---

## Main features

### Training
- PyTorch Lightning training pipeline
- configurable experiments via YAML configs
- support for multiple backbone families such as ResNet and ViT
- MLflow logging for runs, metrics, and artifacts

### Data pipeline
- dataset manifest generation
- deterministic split generation
- dataset validation checks
- shared preprocessing across train/eval/inference workflows

### Evaluation
- offline evaluation pipeline
- explainability workflows for qualitative failure analysis
- foundation for robustness and reliability benchmarking
- extensible metrics and benchmarking structure

### Inference and serving
- reusable inference package
- batch prediction scaffolding
- FastAPI service scaffolding
- monitoring and prediction logging scaffolding

### Engineering
- typed configuration validation
- test structure
- Docker and CI scaffolding
- docs-oriented roadmap toward a reproducible ML product surface

---

## Repository structure

```text
.
├── configs/                 # Training / evaluation / model / MLflow configuration
├── data/                    # Raw, interim, processed data artifacts
├── docker/                  # Training and API container definitions
├── docs/                    # Architecture, roadmap, deployment, benchmarking docs
├── scripts/                 # CLI entrypoints
├── src/
│   ├── config/             # Typed config schema and loading
│   ├── data/               # Manifest building, splitters, validation, datamodule
│   ├── evaluation/         # Metrics, evaluation, robustness-related code
│   ├── explainability/     # Grad-CAM / SHAP / qualitative analysis
│   ├── inference/          # Prediction and model loading logic
│   ├── inference_service/  # FastAPI service scaffolding
│   ├── mlops/              # MLflow registry / promotion workflows
│   ├── monitoring/         # Prediction logging and drift checks
│   ├── models/             # Model definitions and Lightning modules
│   ├── preprocessing/      # Shared transforms
│   └── training/           # Training utilities
├── tests/                   # Unit and integration tests
└── README.md
```

---

## Core design principles

### 1. Data is a first-class artifact
Training should not depend on hidden folder traversal or ad hoc split logic.  
The dataset should be inspectable, reproducible, and versionable through manifests and persisted split artifacts.

### 2. Evaluation goes beyond accuracy
The system should report not only standard classification metrics, but also robustness under corruption, calibration quality, selective prediction behavior, and domain-shift sensitivity.

### 3. Reliability matters operationally
A useful inspection model must communicate confidence, expose failure modes, and support thresholding or abstention in uncertain cases.

### 4. Reproducibility is part of the result
Configs, manifests, runs, and model versions should be traceable through structured artifacts and MLflow logging.

### 5. Serving should reuse the same inference core
Batch inference, interactive demos, and API serving should rely on the same model-loading and preprocessing contracts.

---

## Planned workflow

### Data
- build canonical dataset manifest
- validate dataset integrity and split disjointness
- generate deterministic train/val/test/OOD/robustness splits

### Train
- train Lightning-based classifier
- log metrics and artifacts to MLflow
- compare architectures and data variants

### Benchmark
- run in-distribution evaluation
- run corruption robustness benchmarks
- run OOD and slice-based evaluation
- compute calibration and selective prediction metrics

### Serve
- load model from checkpoint or registry
- expose prediction through batch or FastAPI service
- log predictions and monitor simple drift indicators

### Release
- evaluate candidate model on the standard benchmark suite
- enforce pass/fail gates before promotion

---

## Example commands

These are the intended one-command workflows for the project surface.

### Data preparation
```bash
python -m src.data.build_manifest
python -m src.data.validate_dataset
```

### Training
```bash
python scripts/train_model.py
```

### Evaluation
```bash
python -m src.evaluation.evaluate
```

### Batch inference
```bash
python scripts/batch_predict.py --input path/to/images --output predictions.csv
```

### API serving
```bash
uvicorn src.inference_service.app:app --host 0.0.0.0 --port 8000
```

> Exact commands may evolve as the repository moves toward a more formal job-based interface.

---

## Tech stack

- **Python**
- **PyTorch**
- **PyTorch Lightning**
- **scikit-learn**
- **pandas / NumPy**
- **MLflow**
- **FastAPI**
- **Docker**
- **GitHub Actions**

---

## Project status

SentinelInspect is currently in the transition from:

> reproducible crack classification experiment

to:

> reusable reliability-first visual inspection ML system

Already present:
- Lightning training
- config-driven experimentation
- MLflow tracking
- shared preprocessing
- evaluation and explainability foundations
- early deterministic data tooling
- repository scaffolding for inference, serving, monitoring, and MLOps

Current priorities:
1. make manifest + persisted splits the true source of truth
2. formalize robustness and reliability benchmarking
3. complete reusable inference and service contracts
4. add release gating, monitoring, and stronger CI coverage

See [`docs/roadmap.md`](docs/roadmap.md) for the detailed development plan.

---

## Documentation

Planned and existing documentation includes:

- [`docs/roadmap.md`](docs/roadmap.md)
- `docs/architecture.md`
- `docs/dataset_contract.md`
- `docs/benchmarking.md`
- `docs/deployment.md`
- `docs/model_card.md`

---

## Portfolio positioning

This repository is intended to demonstrate:

- applied ML engineering
- reproducible experimentation
- reliability-aware model evaluation
- data-centric workflow design
- production-facing inference and lifecycle thinking

It is positioned as a stronger portfolio artifact than a standalone image classification notebook or benchmark, because it emphasizes both **research-grade evaluation** and **engineering discipline**.

---

## Author

**Lucas Perrier**  
MSc in Data & Artificial Intelligence, ESILV  
Interests: scientific machine learning, uncertainty-aware modelling, spatio-temporal systems, and robust applied ML

GitHub: [github.com/lucasperrier](https://github.com/lucasperrier)

---