# SentinelInspect

## Production-first visual inspection triage system

SentinelInspect is a computer vision project built to demonstrate how a classification
prototype can be turned into a more production-ready inspection workflow.

The use case is crack classification, but the real goal is broader:

> build a visual inspection triage system with reproducible data, standardized
> evaluation, deployable inference, and basic delivery infrastructure

This repository is a portfolio project for ML Engineer roles, so the emphasis is not only
on model training. The emphasis is on the parts that make a system credible to ship.

See [`docs/architecture.md`](docs/architecture.md) for the system design and
[`docs/roadmap.md`](docs/roadmap.md) for the full phased plan.

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

Many CV projects stop at training accuracy. That is not enough for a strong ML engineering
portfolio. A deployable system also has to answer:

- what exact data was used for training
- are splits deterministic and reproducible
- what artifacts define the evaluation result
- how is inference exposed outside training code
- what happens to low-confidence predictions
- how would the system be tested, packaged, and run

This project focuses on those questions.

---

## What works today

The following components contain real, functioning implementation:

| Area | File(s) | Status |
| --- | --- | --- |
| Manifest generation (paths, labels, dims, SHA256) | `src/data/build_manifest.py` | Implemented |
| Deterministic stratified split generation | `src/data/splitters.py` | Implemented |
| Dataset integrity + split-leakage validation | `src/data/validate_dataset.py` | Implemented |
| Lightning datamodule consuming persisted CSVs | `src/data/datamodule.py` | Implemented |
| Albumentations preprocessing pipelines | `src/preprocessing/transforms.py` | Implemented |
| ResNet-50 / ViT Lightning modules | `src/models/resnet50.py`, `src/models/vit.py` | Implemented |
| Hydra + MLflow training entrypoint | `src/training/train.py` | Implemented* |
| Offline evaluation + artifact writing | `src/evaluation/evaluate.py` | Implemented* |
| Single-image checkpoint inference | `src/inference/predict.py` | Implemented |
| Grad-CAM / SHAP explainability | `src/explainability/` | Implemented |
| Typed config validation (Hydra + Pydantic) | `src/config/schema.py`, `src/config/load.py` | Implemented |
| Data-layer unit tests | `tests/`, `tests/unit/` | Implemented |

`*` The training and evaluation entrypoints are written but the end-to-end run is not yet
green after the recent data-layer refactor. Closing that gap is **Phase 0** of the roadmap
and is the single most important next step.

This is already well beyond a notebook-only project.

---

## What is scaffolding, not yet built

These paths exist as empty placeholder files. They are planned system surfaces, not
completed features:

- **Inference service** — `src/inference_service/` (`app.py`, `routes.py`, `schemas.py`, `dependencies.py`, `logging.py`)
- **Batch + shared inference** — `src/inference/batch_predict.py`, `contracts.py`, `model_loader.py`
- **Evaluation helpers** — `src/evaluation/metrics.py`, `reports.py`
- **Monitoring** — `src/monitoring/` (`prediction_logger.py`, `drift.py`, `reporting.py`)
- **MLOps** — `src/mlops/` (`artifact_store.py`, `promote_model.py`, `registry.py`, `tracking.py`)
- **Offline jobs** — `src/jobs/`
- **Shared utilities** — `src/utils/`
- **Delivery** — `docker/Dockerfile.api`, `docker/Dockerfile.train`, `.github/workflows/ci.yaml`, `pyproject.toml`

Empty files are kept only where imminently planned. Anything not on the active roadmap
should be removed rather than left hollow.

---

## Output contract

The target prediction contract for each image:

- `predicted_label`
- `confidence_score`
- `needs_review`
- `model_metadata` (model name, checkpoint, version)

Status today: predicted label and confidence are implemented in the inference path;
`needs_review` is formalized in Phase 2; metadata/provenance is completed with the service
layer in Phase 3.

---

## Repository structure

```
.
├── configs/            # Hydra config groups: data, model, trainer, mlflow, service
├── data/
│   └── processed/      # manifests/ and splits/ — the dataset contract
├── docker/             # train/api images (to be implemented)
├── docs/               # architecture.md, roadmap.md
├── reports/            # evaluation bundles
├── runs/               # training checkpoints / MLflow run artifacts
├── scripts/            # CLI wrappers (to be implemented)
├── src/
│   ├── config/         # typed config schema + loader
│   ├── data/           # manifest, splitters, validation, datamodule
│   ├── evaluation/     # evaluate.py (+ metrics/reports, planned)
│   ├── explainability/ # Grad-CAM, SHAP
│   ├── inference/      # predict.py (+ batch/shared core, planned)
│   ├── inference_service/  # FastAPI service (planned)
│   ├── models/         # resnet50, vit
│   ├── monitoring/     # prediction logging, drift (planned)
│   ├── preprocessing/  # transforms
│   └── training/       # train.py
├── tests/
├── README.md
└── requirements.txt
```

---

## Roadmap at a glance

Full detail and success criteria in [`docs/roadmap.md`](docs/roadmap.md).

- **Phase 0 — Make the core path executable** *(must ship, first)* — fix the datamodule
  call signature in train/eval, add label encoding, ship a fixture dataset, run end-to-end once.
- **Phase 1 — Data artifacts as the source of truth** *(must ship, largely done)* — route
  remaining paths through config; fail fast on invalid artifacts.
- **Phase 2 — Standardize the evaluation bundle** *(must ship)* — move metric helpers into
  `metrics.py`, save per-sample confidence, add the `needs_review` triage rule.
- **Phase 3 — Shared inference and a usable API** *(must ship)* — one prediction core behind
  CLI, batch, and FastAPI; return the full output contract.
- **Phase 4 — Real packaging, CI, and Docker** *(must ship)* — fill `pyproject.toml`, write a
  real CI workflow, implement both Dockerfiles.
- **Phase 5 — Minimal monitoring hooks** *(nice to have)* — structured prediction logging and
  one offline summary.

---

## Getting started

> Note: a clean end-to-end run is Phase 0 work. The commands below describe the intended
> workflow and become fully reproducible once Phase 0 lands.

```bash
# install
pip install -r requirements.txt

# 1. build the manifest from data/raw/
python -m src.data.build_manifest

# 2. generate deterministic splits
python -m src.data.splitters

# 3. validate dataset integrity and split overlap
python -m src.data.validate_dataset \
  --manifest data/processed/manifests/manifest.csv \
  --train data/processed/splits/train.csv \
  --val data/processed/splits/val.csv \
  --test data/processed/splits/test.csv

# 4. train (Hydra-configured, MLflow-tracked)
python -m src.training.train

# 5. evaluate a checkpoint into reports/
python -m src.evaluation.evaluate checkpoint_path=/path/to/model.ckpt

# 6. single-image inference
python -m src.inference.predict \
  checkpoint_path=/path/to/model.ckpt image_path=/path/to/image.jpg
```