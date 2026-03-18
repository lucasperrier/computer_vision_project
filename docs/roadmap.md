# Roadmap

This roadmap reorients the repository from a crack classification experiment into a **reliability-first visual inspection ML system**.

The goal is not to throw away the current repo shape. The goal is to **keep the working foundations already present**, formalize them, and make the data pipeline and evaluation pipeline the main artifact of the project.

In other words: keep the current training, config, evaluation, explainability, and serving scaffolding, then upgrade the repo so it demonstrates strong ML engineering under real-world constraints such as noisy inputs, limited labels, distribution shift, and deployment readiness.

## Project goal

Transform the repository into a reusable ML system that supports:

- deterministic, auditable dataset construction with manifests, persisted splits, and dataset contracts
- data quality validation and data ablations as first-class workflows
- robustness evaluation under corruptions, artifacts, and domain shift
- reliability analysis with calibration and selective prediction
- data-efficient learning loops such as active learning, pseudo-labeling, and distillation
- reusable batch inference and API serving with monitoring hooks
- MLflow-managed lifecycle with registry, promotion, and evaluation before release
- tests, packaging, CI, and portfolio-grade documentation
- basic scale-aware execution and profiling hooks for training and inference

## What already exists and should stay

The repo already contains useful foundations that should be preserved and extended rather than replaced:

- **PyTorch Lightning training** is already the training backbone.
- **YAML/Hydra-style configuration** already exists under `configs/`.
- **Typed config validation** already exists in `src/config/schema.py` and `src/config/load.py`.
- **Shared preprocessing** already exists in `src/preprocessing/transforms.py`.
- **Offline evaluation and explainability** already exist in `src/evaluation/` and `src/explainability/`.
- **Early deterministic data utilities** already exist in `src/data/build_manifest.py`, `src/data/splitters.py`, and `src/data/validate_dataset.py`.
- **Inference / service / monitoring / MLOps scaffolding** already exists in `src/inference/`, `src/inference_service/`, `src/monitoring/`, and `src/mlops/`.
- **Docker, CI, and test structure** already exist in the repo layout, even though some pieces are still placeholders.

## Current alignment summary

The repo is **partially aligned already** with the new direction.

### Already aligned

- deterministic manifest builder exists
- deterministic split generation exists
- dataset validation exists
- preprocessing has already been extracted into a shared module
- typed config validation is already in place
- evaluation and explainability workflows already produce reusable artifacts
- directories and modules for inference, API serving, monitoring, MLOps, jobs, and CI are already present

### Partially aligned

- `src/data/datamodule.py` still performs runtime split generation instead of fully consuming persisted split artifacts
- manifest output is currently CSV/JSON-oriented rather than centered on a canonical parquet contract
- robustness evaluation exists as a concept, but not yet as a complete corruption benchmark suite with tracked degradation curves
- inference/service/monitoring/registry modules exist, but several files are placeholders or only minimally implemented
- tests exist, but current coverage is still narrow and one current test relies on the real dataset layout
- Dockerfiles and CI workflow files exist, but are not yet fully implemented

### Missing or not yet formalized

- dataset contract documentation and richer quality reporting
- data ablation job orchestration
- reliability metrics such as ECE, Brier score, and selective prediction curves
- slice-based and OOD-first evaluation workflow
- active learning, pseudo-labeling, and distillation loops
- release gating tied to MLflow promotion
- production-ready API contracts, metrics exposure, and prediction logging
- explicit throughput/latency profiling and scale-oriented execution options

## Reoriented roadmap

The roadmap below keeps the current foundations and focuses on **formalizing and extending what is already there**.

---

## Phase 1 — Deterministic data and dataset contract

### 1. Canonical dataset manifest

**Keep and extend:** `src/data/build_manifest.py`

Current state:

- image enumeration already exists
- file hashing already exists
- metadata fields such as dataset, label, size, and dimensions already exist

What to improve:

- make the manifest the single source of truth for downstream training/evaluation
- add optional EXIF / richer source metadata where useful
- produce `manifest.parquet` as the canonical artifact, with CSV as optional export
- ensure schema is explicit and documented as a dataset contract

Target outputs:

- `data/processed/manifests/manifest.parquet`
- `data/processed/manifests/manifest.csv` (optional convenience export)

Success criteria:

- every sample is represented exactly once
- training never relies on raw folder traversal as its hidden source of truth
- dataset contents are inspectable, diffable, and reproducible

### 2. Deterministic splits and robustness split definitions

**Keep and extend:** `src/data/splitters.py`

Current state:

- deterministic train/val/test split generation already exists
- `robustness.csv` output already exists

What to improve:

- add `ood.csv` for held-out source/domain evaluation
- define the logic for robustness and OOD slices from the manifest itself
- remove runtime split generation from `CrackDataModule.setup()` and load persisted split membership instead

For this project, define OOD concretely:

- **OOD = train on SDNET and evaluate on CCIC held out entirely**
- **OOD = train on CCIC and evaluate on SDNET held out entirely**

This makes domain-shift evaluation explicit and grounded in the actual datasets used by the repository.

Target outputs:

- `train.csv`
- `val.csv`
- `test.csv`
- `ood.csv`
- `robustness.csv`

Success criteria:

- split membership is fixed and auditable
- no hidden randomness remains in the datamodule
- OOD and robustness evaluation become part of the default workflow

### 3. Dataset validation and quality report

**Keep and extend:** `src/data/validate_dataset.py`

**Add:** `src/data/data_quality_report.py`

Current state:

- unreadable/corrupt file checks already exist
- duplicate row/path checks already exist
- split overlap checks already exist
- class balance reporting already exists

What to improve:

- add exact duplicate detection based on content hash at report level
- add near-duplicate detection with perceptual hashing
- add resolution buckets, blur proxy, and compression artifact proxies
- produce a reusable report artifact saved alongside manifests/splits

Success criteria:

- training fails before starting if data is invalid
- quality reporting becomes a standard run artifact
- duplicates and leakage are systematically prevented

---

## Phase 2 — Data-centric experimentation

### 4. Data ablation harness

**Add:** `src/jobs/run_data_ablations.py`

Purpose:

- treat data composition as a controlled experiment, not a hidden preprocessing detail

Responsibilities:

- derive dataset variants from the canonical manifest
- remove duplicates / near-duplicates
- filter low-quality samples
- rebalance by class or source
- compare source mixtures
- train and evaluate each variant under the same model configuration
- log comparison runs to MLflow

Minimum target:

- implement at least **3 to 5 meaningful ablations**
- publish a compact comparison table showing metric deltas across variants

Success criteria:

- you can answer which data changes improved robustness or generalization
- experiments are reproducible and directly comparable
- the workflow runs as a job, not as manual notebook work

---

## Phase 3 — Robustness and reliability evaluation

### 5. Corruption and noise benchmarks

**Keep and extend:** `src/evaluation/robustness.py`

**Add:** `src/evaluation/corruptions.py`, `src/jobs/run_robustness_benchmark.py`

Current state:

- a robustness module already exists, but it is not yet a full benchmark system

What to add:

- synthetic corruptions such as blur, JPEG artifacts, sensor noise, brightness/contrast shifts, and crops
- degradation curves across corruption severities
- MLflow logging for robustness metrics and artifacts

Success criteria:

- the repo quantifies sensitivity to realistic artifacts
- robustness becomes a tracked metric rather than a side analysis

### 6. Reliability metrics and risk-aware evaluation

**Add:** `src/evaluation/reliability.py`

**Extend:** `src/evaluation/metrics.py`

Current state:

- `src/evaluation/metrics.py` exists but is still effectively a placeholder for the broader release-evaluation role

What to add:

- expected calibration error (ECE)
- Brier score
- confidence histograms
- reliability diagrams
- selective prediction curves such as coverage vs error
- threshold recommendations for deployment

Success criteria:

- deployment thresholds can be justified quantitatively
- the system can express when not to trust the model

### 7. Slice-based evaluation and OOD generalization

**Add:** `src/evaluation/slices.py`, `src/jobs/run_slice_eval.py`

Responsibilities:

- evaluate by slice such as source dataset, resolution bucket, crack size proxy, and background texture proxy
- evaluate explicitly on `ood.csv`
- generate a standardized evaluation report artifact

Success criteria:

- performance claims become more nuanced and defensible
- OOD behavior is visible and measurable

---

## Phase 4 — Data-efficient learning loop

### 8. Active learning loop

**Add:** `src/learning/active_learning.py`, `src/jobs/run_active_learning.py`

Responsibilities:

- start from a small labeled seed set
- iteratively train, score the unlabeled pool, acquire samples, and update the labeled set
- support uncertainty-based or disagreement-based selection
- log performance vs label budget curves

Success criteria:

- the repo demonstrates competence under limited-label settings
- performance can be shown as a function of labeling cost

### 9. Pseudo-labeling and distillation

**Add:** `src/learning/pseudo_label.py`, `src/learning/distill.py`

Responsibilities:

- generate pseudo-labels from a teacher model with confidence filtering
- train a student under mixed supervision
- distill toward a smaller deployment-oriented backbone

Success criteria:

- the repo demonstrates practical semi-supervised learning and model compression
- robustness improvement or inference-cost reduction is measurable

---

## Phase 5 — Shared preprocessing and structured config

### 10. Single preprocessing definition used everywhere

**Keep and harden:** `src/preprocessing/transforms.py`

Current state:

- shared train / eval / inference transform builders already exist
- `src/data/datamodule.py` already uses preprocessing helpers

What to improve:

- ensure evaluation and inference codepaths all consume this shared module consistently
- version preprocessing through config
- add tests for transform defaults and parity across workflows
- optionally add robustness-specific preprocessing toggles without introducing training/serving skew

Success criteria:

- there is no transform duplication across train/eval/inference
- preprocessing is versioned and testable
- training/serving skew is reduced

### 11. Config composition and validation

**Keep and extend:** `configs/` structure, `src/config/schema.py`, `src/config/load.py`

Current state:

- structured config directories already exist
- typed validation already exists through Pydantic models

What to improve:

- keep the config tree clean and task-oriented
- add benchmark-specific configs such as `configs/benchmarks/robustness.yaml`
- tighten schema coverage for release evaluation, inference, monitoring, and distributed execution
- fail fast on invalid benchmark/service/runtime configs

Success criteria:

- swapping model/data/benchmark settings is trivial
- invalid configs fail early and clearly
- config composition remains a visible strength of the repo

---

## Phase 6 — Inference and serving

### 12. Reusable inference core and standardized outputs

**Keep and complete:** `src/inference/model_loader.py`, `src/inference/predict.py`, `src/inference/contracts.py`

Current state:

- the inference package exists
- `predict.py` currently performs single-image inference but still carries script-level logic rather than a reusable core
- `model_loader.py` exists as a placeholder

What to improve:

- isolate model loading from CLI/script concerns
- support checkpoint and MLflow-registry loading
- standardize prediction outputs with label, confidence, optional uncertainty, model version, and preprocessing version

Success criteria:

- inference is decoupled from training/evaluation scripts
- both batch and API serving share the same inference core
- outputs are stable and contract-driven

### 13. Batch inference

**Keep and complete:** `src/inference/batch_predict.py` and/or `scripts/batch_predict.py`

Current state:

- batch-prediction entrypoint names already exist, but the implementation is incomplete

What to improve:

- support folder, CSV, and manifest inputs
- produce CSV/parquet outputs with timestamp, model version, and preprocessing metadata
- optionally save explainability artifacts for a sampled subset

Success criteria:

- the repo supports practical offline scoring and audit workflows
- outputs are structured and traceable

### 14. FastAPI service and metrics

**Keep and complete:** `src/inference_service/app.py`, `routes.py`, `schemas.py`

Current state:

- the service package structure already exists
- the core app/routes/schemas files are still placeholders

What to deliver:

- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch`
- `GET /metrics` for Prometheus-style metrics

Success criteria:

- the repository exposes a deployable demo service
- request/response contracts are typed and stable
- latency and service behavior are observable

---

## Phase 7 — Lifecycle, monitoring, release gates, and scale proof points

### 15. MLflow registry and promotion workflow

**Keep and complete:** `src/mlops/registry.py`, `src/mlops/promote_model.py`

Current state:

- the MLOps package exists
- registry/promotion files exist but are not yet implemented end-to-end

What to improve:

- register models with signatures and input examples
- promote only when evaluation gates pass
- point serving/deployment to registry-managed versions

Success criteria:

- MLflow becomes a lifecycle tool, not just an experiment tracker
- promotion decisions become explicit and auditable

### 16. Prediction logging and drift checks

**Keep and complete:** `src/monitoring/prediction_logger.py`, `src/monitoring/drift.py`

Current state:

- the monitoring package exists
- prediction logging and drift files are present but unfinished

What to add:

- structured logging of latency, confidence, input hash, and model version
- drift checks against training-manifest-derived reference statistics such as resolution, brightness proxy, and blur proxy
- simple alerts as metrics, structured logs, or report artifacts

Success criteria:

- operational awareness is visible in the repo
- data shift risk becomes measurable rather than implied

### 17. Evaluation as a release gate

**Add:** `src/jobs/run_release_evaluation.py`

**Extend:** `src/evaluation/metrics.py`

Responsibilities:

- run the standard benchmark suite: in-distribution test, OOD test, corruption benchmarks, and reliability metrics
- produce one release report artifact
- enforce pass/fail thresholds before promotion

Success criteria:

- release candidates must pass a defined benchmark suite
- regressions are caught before deployment or promotion

### 18. Scale-aware execution and profiling

**Add:** distributed/runtime support and profiling notes in training and serving configs

What to add:

- optional `torchrun` execution path
- optional FSDP/distributed configuration hooks where appropriate
- basic throughput profiling for training
- basic latency profiling for batch inference and API serving

Success criteria:

- the repo shows awareness of scale-oriented execution, even without large-cluster compute
- throughput and latency are measured rather than assumed
- systems-oriented reviewers can see an explicit performance engineering surface

---

## Phase 8 — Quality, reproducibility, and delivery

### 19. Tests with fixtures

**Keep and expand:** `tests/`

Current state:

- test structure already exists
- `tests/test_datamodule.py` currently depends on the real dataset and old datamodule behavior

What to improve:

- move toward fast fixture-driven tests
- add unit tests for manifest building, splitters, transforms, metrics, and model loading
- add integration tests for API and batch inference
- add a smoke training test on a tiny synthetic dataset

Success criteria:

- tests run in CI without the full raw dataset
- core logic is covered by fast automated checks

### 20. Containerization for training and serving

**Keep and complete:** `docker/Dockerfile.train`, `docker/Dockerfile.api`

Current state:

- Dockerfiles already exist as placeholders

What to improve:

- implement reproducible training and serving images
- align with pinned dependency strategy
- support easy demo and local deployment

Success criteria:

- builds are reproducible
- containerized demo paths are simple and credible

### 21. CI pipeline

**Keep and complete:** `.github/workflows/ci.yaml`

Current state:

- CI workflow file already exists as a placeholder

What to include:

- lint / format checks
- unit and integration tests
- container build checks
- optional tiny smoke benchmark or release-gate subset

Optional:

- pre-commit hooks

Success criteria:

- regressions are automatically caught
- engineering discipline is visible to reviewers

---

## Phase 9 — Portfolio packaging

### 22. Documentation that sells the system

**Keep and expand:** `README.md`, `docs/`

Current state:

- a usable README already exists
- architecture/serving docs already exist in `docs/`

What to improve:

- reposition the project as a reliability-first visual inspection system
- add or update:
  - `docs/architecture.md`
  - `docs/dataset_contract.md`
  - `docs/benchmarking.md`
  - `docs/deployment.md`
  - `docs/model_card.md`
- document one-command workflows for data, train, benchmark, and serve
- include diagrams for data flow, training flow, release flow, and monitoring flow

Success criteria:

- a reviewer understands the system in five minutes
- the repository reads like an ML product with research-grade evaluation, not just a model experiment

---

## Recommended implementation order

If time is limited, the best order is:

1. finish manifest + persisted splits + validation as the true source of truth
2. refactor the datamodule to consume persisted split artifacts
3. add robustness benchmark + reliability metrics
4. add data ablation harness with at least 3–5 concrete variants
5. complete reusable inference core + batch inference
6. complete FastAPI service + metrics + prediction logging
7. add MLflow registry + release evaluation gate
8. add basic torchrun/FSDP config hooks and throughput/latency profiling
9. finish tests + Docker + CI
10. rewrite README/docs for portfolio packaging

## Final target state

By the end, the repository should support:

- deterministic, auditable data ingestion
- dataset quality checks and data ablations as first-class workflows
- robustness, OOD, and reliability evaluation beyond accuracy
- data-efficient learning loops for limited-label settings
- batch and online inference with monitoring
- MLflow-managed lifecycle with release gates
- reproducible packaging, CI enforcement, and strong project documentation
- basic scale-aware training and inference profiling

## Positioning

The finished repo should present itself as:

**a reliability-first visual inspection ML system built for real-world noise, limited labels, and production-facing evaluation**

That positioning keeps the current crack-detection domain, but upgrades the project from “model training experiment” into a stronger applied ML engineering and research portfolio piece.