# SentinelInspect

## Production-first visual inspection triage system

SentinelInspect is a computer vision project built to demonstrate how a classification prototype can be turned into a more production-ready inspection workflow.

The use case is crack classification, but the real project goal is broader:

> build a visual inspection triage system with reproducible data, standardized evaluation, deployable inference, and basic delivery infrastructure

This repository is designed as a portfolio project for ML Engineer roles, so the emphasis is not only on model training. The emphasis is on the parts that make a system credible to ship.

---

## What this project is

SentinelInspect is a **decision-support system** for inspection workflows.

The intended behavior is:

1. ingest an image
2. predict `crack` or `no_crack`
3. attach a confidence score
4. route uncertain cases to manual review
5. log results for later analysis

This is a triage framing, not a claim of full inspection autonomy.

---

## Why this project exists

Many CV projects stop at training accuracy.

That is not enough for a strong ML engineering portfolio.

A real deployable system also needs to answer questions such as:

- what exact data was used for training
- are splits deterministic and reproducible
- what artifacts define the evaluation result
- how is inference exposed outside training code
- what happens to low-confidence predictions
- how would the system be tested, packaged, and run

This project focuses on those questions.

---

## Current scope

The current target is a **shippable v1**, not a full MLOps platform.

The priority is to finish a small set of high-signal capabilities well:

- persisted data manifests and splits as the source of truth
- standardized evaluation bundles
- shared inference for single-image, batch, and API use
- Dockerized execution and CI
- minimal prediction logging / monitoring hooks

That scope is deliberate. It is the shortest path to a repo that feels production-minded rather than aspirational.

---

## What is implemented today

The following parts are already real and present in the repository:

- manifest generation in `src/data/build_manifest.py`
- deterministic split generation in `src/data/splitters.py`
- dataset validation in `src/data/validate_dataset.py`
- shared preprocessing in `src/preprocessing/`
- Hydra-configured training in `src/training/train.py`
- MLflow tracking configuration in `configs/mlflow/default.yaml`
- offline evaluation in `src/evaluation/evaluate.py`
- explainability utilities in `src/explainability/`
- single-image inference in `src/inference/predict.py`
- tests covering core data and preprocessing components

This is already more than a notebook-only project.

---

## What is still in progress

Several directories and files exist as scaffolding but are not yet finished:

- `src/evaluation/metrics.py`
- `src/inference/batch_predict.py`
- `src/inference_service/`
- `src/monitoring/`
- `src/mlops/`
- `src/jobs/`
- `.github/workflows/ci.yaml`
- `docker/Dockerfile.api`
- `docker/Dockerfile.train`
- `pyproject.toml`
- some wrappers in `scripts/`

These should be read as planned system surfaces, not as completed features.

---

## Core system design

The intended system flow is:

### 1. Data contract
Raw image data is inventoried into a manifest and assigned to persisted splits.

### 2. Reproducible training
Training runs from config, logs to MLflow, and produces checkpoints.

### 3. Standardized evaluation
Each candidate model produces a consistent evaluation bundle with metrics and prediction artifacts.

### 4. Deployable inference
A shared prediction core serves local inference, batch workflows, and an API surface.

### 5. Review-aware operation
Low-confidence predictions can be flagged for review through a `needs_review` rule.

### 6. Basic observability
Predictions can be logged and summarized for operational analysis.

That is the production-first interpretation of this repository.

---

## Output contract

The target prediction contract for each image is:

- `predicted_label`
- `confidence_score`
- `needs_review`
- `model_metadata`

At the moment:

- predicted label is implemented
- confidence is available in the inference path
- `needs_review` is a target capability still being formalized
- metadata / provenance should be expanded as the service layer is completed

---

## Evaluation posture

This repo is intentionally evaluation-first.

The project is not meant to say “train a model and hope.”  
It is meant to say “train a model, evaluate it in a repeatable way, and define how it would be used.”

Artifacts already written by `src/evaluation/evaluate.py` include outputs such as:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.npy`
- `predictions.npz`

The next step is to turn these into a cleaner release bundle that also supports confidence-aware triage.

---

## Repository structure

```text
.
├── configs/
│   ├── data/
│   ├── mlflow/
│   ├── model/
│   ├── service/
│   ├── trainer/
│   ├── eval.yaml
│   ├── inference.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── features/
│       ├── manifests/
│       └── splits/
├── docker/
├── docs/
│   ├── archiecture.md
│   ├── roadmap.md
│   └── serving_design.md
├── reports/
├── runs/
├── scripts/
├── src/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   ├── inference_service/
│   ├── jobs/
│   ├── mlops/
│   ├── models/
│   ├── monitoring/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
├── tests/
├── README.md
├── requirements.txt
└── mlflow.db