# Roadmap

This roadmap outlines the step-by-step plan to evolve this repository from a research-focused concrete crack classification project into a reusable, production-oriented computer vision classification system.

## Project goal

The objective is to transform the repo into a modular ML engineering project that supports:

- deterministic and reproducible data preparation
- reusable preprocessing across training, evaluation, and inference
- structured configuration management
- deployable inference and API serving
- model lifecycle management with MLflow
- monitoring and release-oriented evaluation
- proper testing, packaging, and CI
- portfolio-grade documentation

---

## Current direction

The repository already has a solid research base with:

- PyTorch Lightning training
- YAML-based experiment configs
- MLflow tracking
- offline evaluation
- explainability workflows

The next phase is to improve the engineering surface so the project demonstrates ML system design and operational maturity, not just experimentation.

---

## Phase 1 — Deterministic data layer

### 1. Create deterministic dataset manifests

Add:

- `src/data/build_manifest.py`

Responsibilities:

- enumerate every image in the raw dataset
- capture source dataset, label, file path, file hash, width, and height
- produce a canonical dataset manifest

Output location:

- `data/processed/manifests/`

Expected artifact examples:

- `data/processed/manifests/manifest.csv`
- `data/processed/manifests/manifest.parquet`

Success criteria:

- every training image is represented in a single canonical manifest
- dataset contents are inspectable and reproducible
- raw folder traversal is no longer the hidden source of truth

A manifest is a table that lists every image and its metadata, in the goal of making data ingestion a reproducible process. Without a manifest, you do not know exactly which files were used, you cannot indpect the whole dataset easily, debugging and reproducibility are weaker. With a manifest, every image is recorder, metadata is explicit, dataset changes become trackable. A manifest is an inventory sheet for your dataset.

---

### 2. Generate and persist split files

Splitters.py takes the canonical manifest and assignes each sample to train, val and test in a deterministic way, then writes fixed csv files. Splitting the dataset this way matters because it ensures no random runtime split drift, the same manifest ad same seed implies the same split files, split membership is auditable and reproducible.

Add:

- `src/data/splitters.py`

Responsibilities:

- generate deterministic dataset splits from the manifest
- persist:
  - `train.csv`
  - `val.csv`
  - `test.csv`
  - `robustness.csv`

Refactor:

- remove runtime split generation from `CrackDataModule.setup()`

Success criteria:

- train/val/test membership is fixed and auditable
- rerunning the repo on the same manifest yields the same split files
- evaluation no longer depends on runtime sampling behavior

---

### 3. Add dataset validation job

validate_dataset.py validates the dataset in the sense that it fails on corruption, missing files, duplicates, and split leakage before training starts. 

Add:

- `src/data/validate_dataset.py`

Responsibilities:

- check for duplicates
- detect unreadable files
- detect corrupt images
- verify split overlap does not exist
- inspect class balance
- fail early before training begins

Success criteria:

- broken or inconsistent data is detected before training
- split leakage is automatically caught
- dataset quality checks become part of the normal workflow

---

## Phase 2 — Shared preprocessing and structured configuration

### 4. Refactor preprocessing into shared modules

transforms.py makes the preprocessing a shared component instead of hidden logic inside training code. It is defined once and reused in datamodule.py, evaluate.py and predict.py. Preprocessing parameters live in config so they are versioned and reproducible. Preprocessing data with transforms.py matters because it prevents training/serving skew, improves reprodicibility, reduces duplication, improves maintainability, enables testing. 

Add:

- `src/preprocessing/transforms.py`

Refactor:

- move Albumentations pipelines out of `datamodule.py`

Responsibilities:

- define canonical preprocessing functions for:
  - training
  - validation
  - evaluation
  - inference

Goal:

- use the same preprocessing logic in training, evaluation, and inference

Success criteria:

- no transform duplication across scripts
- preprocessing becomes reusable and versionable
- training/inference skew risk is reduced

---

### 5. Upgrade config system

Refactor:

- restructure configs into composable sections
- stop manually loading YAML throughout the codebase
- add typed config validation for trainer, data, model, and runtime args

Suggested structure:

- `configs/train.yaml`
- `configs/eval.yaml`
- `configs/inference.yaml`
- `configs/model/resnet50.yaml`
- `configs/model/vit.yaml`
- `configs/data/default.yaml`
- `configs/trainer/default.yaml`
- `configs/mlflow/default.yaml`

Goal:

- support cleaner configuration composition
- improve reproducibility and maintainability
- validate config correctness earlier

Success criteria:

- config logic is centralized
- model/data/runtime choices are easier to swap
- invalid configs fail fast with clear errors

---

## Phase 3 — Inference system

### 6. Create a reusable inference core

Add:

- `src/inference/model_loader.py`
- `src/inference/predict.py`

Responsibilities:

- load models from checkpoints or registry
- support CPU/GPU-aware loading
- expose a clean prediction interface
- return standardized prediction objects

Success criteria:

- inference no longer depends on evaluation scripts
- model loading is reusable outside training code
- single-image predictions can be produced from a stable interface

---

### 7. Build batch inference

Add:

- `src/inference/batch_predict.py`

Responsibilities:

- support folder input
- support CSV/manifest input
- save predictions as CSV or parquet
- include timestamps and model version metadata

Success criteria:

- the repo supports practical offline prediction workflows
- outputs are structured and traceable
- batch prediction can be reused for demos and operations

---

## Phase 4 — Serving layer

### 8. Create FastAPI serving layer

Add:

- `src/inference_service/app.py`

Expose endpoints:

- `GET /health`
- `GET /ready`
- `POST /predict`
- `POST /predict/batch`
- `GET /model-info`

Requirements:

- use Pydantic request/response schemas
- use the reusable inference core under the hood
- validate inputs before prediction

Success criteria:

- the project becomes deployable as a service
- predictions are exposed through a clean API contract
- the repo becomes demoable in interviews

---

## Phase 5 — Model lifecycle and release process

### 9. Integrate MLflow model registry

Add:

- `src/mlops/registry.py`

Refactor:

- update `train.py` to log model artifacts with signatures
- add promotion logic for staging and production

Responsibilities:

- register trained models
- version models consistently
- support promotion workflows

Success criteria:

- MLflow is used for more than experiment logging
- model lifecycle becomes explicit
- deployment can point to a registry-managed model version

---

### 10. Add prediction logging and monitoring

Add:

- `src/monitoring/prediction_logger.py`

Responsibilities:

- persist structured inference logs
- capture metadata such as:
  - timestamp
  - model version
  - confidence
  - latency
  - input hash
- add drift checks comparing live input statistics against training manifest statistics

Success criteria:

- prediction events are observable
- post-deployment monitoring becomes possible
- the project shows awareness of model drift and operational ML risks

---

### 11. Refactor evaluation into release checks

Add:

- `src/evaluation/metrics.py`
- `src/jobs/run_offline_evaluation.py`

Responsibilities:

- centralize metric computation
- make evaluation reproducible and job-based
- treat evaluation as a formal release gate before model promotion

Goal:

- move from one-off evaluation scripts to release-oriented validation

Success criteria:

- release candidates must pass offline evaluation before promotion
- evaluation logic is reusable and better organized
- metrics become a formal part of the model lifecycle

---

## Phase 6 — Software quality and reproducibility

### 12. Expand test suite

Refactor:

- replace data-dependent testing with fixtures

Add tests for:

- manifest building
- preprocessing
- metrics
- model loading
- API behavior
- tiny training smoke tests

Suggested structure:

- `tests/unit/test_manifest_builder.py`
- `tests/unit/test_preprocessing.py`
- `tests/unit/test_metrics.py`
- `tests/unit/test_model_loader.py`
- `tests/integration/test_predict_api.py`
- `tests/integration/test_train_smoke.py`

Success criteria:

- tests run without depending on the full raw dataset
- core logic is covered by unit tests
- inference and training workflows have smoke-level integration coverage

---

### 13. Containerize training and serving

Add:

- `docker/Dockerfile.train`
- `docker/Dockerfile.api`
- `.dockerignore`

Also:

- freeze dependencies with a lockfile strategy

Goal:

- make training and serving reproducible across machines
- prepare the project for deployment and CI build validation

Success criteria:

- training and API can run in reproducible containers
- environment drift is reduced
- the repo becomes much easier to demo and share

---

### 14. Add CI pipeline

Add:

- `.github/workflows/ci.yaml`

CI should run:

- lint
- unit tests
- integration tests
- container build checks

Optional:

- pre-commit hooks

Success criteria:

- every change is automatically checked
- the repo demonstrates engineering discipline
- software quality is enforced continuously rather than manually

---

## Phase 7 — Documentation and portfolio positioning

### 15. Add portfolio-grade documentation

Update:

- `README.md`

Add content for:

- architecture diagram
- training flow
- deployment flow
- monitoring flow
- demo instructions

Also add:

- a model card
- a serving example

Suggested docs:

- `docs/roadmap.md`
- `docs/architecture.md`
- `docs/dataset_contract.md`
- `docs/deployment.md`
- `docs/model_card.md`

Success criteria:

- the project is easy to understand from the repository alone
- the engineering story is visible to recruiters and collaborators
- the repo reads like a real ML system, not just an experiment folder

---

## Recommended implementation order

If time is limited, prioritize in this order:

1. deterministic manifest and split pipeline
2. shared preprocessing module
3. reusable inference core
4. FastAPI serving layer
5. MLflow model registry integration
6. prediction logging and drift checks
7. evaluation as release gate
8. test suite expansion
9. Docker packaging
10. CI pipeline
11. documentation upgrades

---

## Final target state

By the end of this roadmap, the repository should support:

- reproducible dataset preparation
- configurable training across image classification datasets
- shared preprocessing across all stages
- offline and batch inference
- deployable API serving
- model registration and promotion
- evaluation-driven releases
- basic monitoring and drift awareness
- reproducible packaging
- automated quality checks
- strong documentation for portfolio and collaboration

The final project should be positioned not as a single crack detection experiment, but as a reusable computer vision classification system with production-oriented ML engineering design.