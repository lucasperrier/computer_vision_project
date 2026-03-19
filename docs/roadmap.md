# Roadmap

## Production-first inspection triage roadmap

This roadmap reframes the repository as a **production-first visual inspection triage system** rather than a standalone crack-classification experiment.

The governing question is:

> How do we industrialise a computer vision prototype into a deployable, evaluation-first, monitorable inspection workflow that can reduce review workload while controlling miss risk?

The roadmap is grounded in the repo state inspected on this branch. It distinguishes clearly between:

- **current state**: what is implemented today
- **next concrete tasks**: actionable work mapped to real files or directories
- **success criteria**: artifacts and measurable outcomes expected for each capability

This aligns with recurring ML Engineer / MLOps job-offer requirements:

- prototype -> production performance
- CI/CD for ML
- lifecycle orchestration
- benchmark-driven release decisions
- deployable inference
- monitoring and drift management
- latency / cost awareness
- governance and auditability

---

## Phase A — Deterministic data contract

Goal: make the dataset a versioned, auditable contract rather than an implicit folder traversal side effect.

### Current state

Implemented today:

- `src/data/build_manifest.py` builds a manifest from `data/raw/`
- manifest records include path, relative path, dataset, split hint, label, size, dimensions, channels, file size, and SHA256 hash
- `src/data/splitters.py` creates deterministic `train.csv`, `val.csv`, `test.csv`, and `robustness.csv`
- `src/data/validate_dataset.py` validates required columns, duplicates, missing files, corrupt images, and split overlap
- configuration paths already exist in `configs/data/default.yaml`

Limitations today:

- canonical artifact is CSV/JSON, not a fully formalized contract with version semantics
- no explicit `ood.csv` split exists yet
- training still reads raw folders through `src/data/datamodule.py` instead of consuming persisted split files
- no standalone data quality report artifact beyond console validation output

### Next concrete tasks

- update `src/data/datamodule.py` to optionally consume `data/processed/manifests/manifest.csv` and `data/processed/splits/*.csv`
- extend `src/data/splitters.py` to emit `ood.csv` for domain-shift evaluation
- extend `src/data/validate_dataset.py` or add a companion report module under `src/data/` for richer quality summaries
- document the contract fields in `docs/archiecture.md` once that doc is no longer a placeholder
- promote `configs/data/default.yaml` as the single path source for manifest/split locations

### Success criteria

- training and evaluation can run from persisted manifest/split artifacts without hidden runtime splitting
- dataset lineage is inspectable via saved manifest and split files
- validation failures stop downstream training jobs
- OOD membership is explicit and reproducible

---

## Phase B — Training as a reproducible job

Goal: move from experiment scripts to an industrialised, reproducible training job.

### Current state

Implemented today:

- `src/training/train.py` is a Hydra entrypoint
- configs are structured under `configs/`
- deterministic seeding is enabled via `seed_everything` when configured
- PyTorch Lightning `Trainer` is used for training, validation, and test passes
- MLflow tracking is wired through `configs/mlflow/default.yaml`
- best checkpoint export is implemented under `trained_models/...`

Limitations today:

- top-level wrappers in `scripts/train_model.py` are placeholders
- `pyproject.toml` is empty, so packaging and reproducible install metadata are incomplete
- no formal job orchestration layer exists in `src/jobs/` yet
- data loading still depends on runtime split generation in `CrackDataModule`

### Next concrete tasks

- keep `src/training/train.py` as the canonical job entrypoint and wire stable wrappers only after the core contract is fixed
- update `src/training/reproducibility.py` and related helpers if more reproducibility metadata is needed
- make the training log explicit data contract metadata: manifest path, split directory, seed, config name, checkpoint path
- turn placeholder `scripts/train_model.py` into a thin wrapper only after the internal interface stabilizes
- populate `pyproject.toml` to support reproducible installs and CI execution

### Success criteria

- one documented training command works reliably from config
- MLflow contains params, metrics, and checkpoint lineage for each run
- rerunning the same config and seed reproduces equivalent artifacts and split membership
- training consumes versioned data artifacts rather than hidden folder splits

---

## Phase C — Evaluation pipeline and release gates

Goal: make evaluation the main release decision surface, not a post-hoc notebook exercise.

### Current state

Implemented today:

- `src/evaluation/evaluate.py` runs offline test evaluation
- saved artifacts include `metrics.json`, `classification_report.txt`, `confusion_matrix.npy`, and `predictions.npz`
- `reports/eval_pretrained_*` and `reports/eval_trained_*` contain historical outputs
- `src/explainability/` provides Grad-CAM / SHAP-oriented analysis utilities
- `src/evaluation/robustness.py` contains helper functions for localization accuracy and faithfulness-drop style metrics

Limitations today:

- `src/evaluation/metrics.py` is a placeholder
- no formal regression benchmark against a previous champion model
- no OOD job, no corruption severity benchmark, no calibration metrics, no selective prediction metrics
- release gates are conceptual only and not yet enforced by CI or registry promotion

### Next concrete tasks

- implement standardized metric helpers in `src/evaluation/metrics.py`
- add evaluation job modules under `src/jobs/` for offline benchmark execution instead of manual orchestration
- extend `src/evaluation/evaluate.py` to emit richer provenance fields and gate-ready summary output
- add OOD evaluation support once `ood.csv` exists
- document a candidate release policy in docs and later tie it to CI/CD and promotion logic

### Success criteria

- every candidate model produces a standardized evaluation bundle
- evaluation compares candidate vs reference model with explicit pass/fail outcomes
- gate dimensions include core quality, robustness, calibration, and selective prediction
- release decisions become reproducible artifacts, not ad hoc discussion

---

## Phase D — Robustness, OOD, calibration, and selective prediction

Goal: support risk-aware inspection triage under noise and domain shift.

### Current state

Implemented today:

- robustness is recognized as a first-class concern in module structure and historical reports
- `robustness.csv` split artifact is generated by `src/data/splitters.py`
- explainability tooling exists for qualitative failure analysis

Limitations today:

- `robustness.csv` is currently a copy of the split dataframe, not a dedicated corruption benchmark definition
- no synthetic corruption benchmark runner exists
- no expected calibration error, Brier score, reliability diagrams, or coverage-risk curves are implemented
- `needs_review` / abstention remains a target behavior rather than a shipped feature

### Next concrete tasks

- define corruption benchmark utilities under `src/evaluation/` using the existing module structure
- add calibration and selective prediction metrics to `src/evaluation/metrics.py`
- extend prediction outputs so confidence can drive an explicit `needs_review` policy
- persist threshold recommendations in evaluation artifacts for operating point selection
- use `reports/` as the canonical location for richer benchmark outputs

### Success criteria

- the repo can report how performance changes under blur, compression, noise, and illumination shifts
- evaluation artifacts include calibration and coverage-risk metrics
- the triage contract supports a configurable review threshold
- operating point selection can be justified quantitatively

---

## Phase E — Deployment surface: shared inference core, batch, and FastAPI

Goal: expose a deployable inference surface with stable contracts and minimal training-serving skew.

### Current state

Implemented today:

- `src/inference/predict.py` performs single-image inference from a checkpoint
- inference uses shared config loading via `src/config/load.py`
- `configs/inference.yaml` and `configs/service/default.yaml` define configuration surfaces

Limitations today:

- `src/inference/batch_predict.py` is a placeholder
- `src/inference_service/app.py`, `routes.py`, and `schemas.py` are placeholders
- the `scripts/` wrappers for batch and API serving are placeholders
- no stable public request/response schema is implemented yet

### Next concrete tasks

- extract the reusable prediction core from `src/inference/predict.py` so both API and batch use the same function path
- implement typed request/response schemas in `src/inference_service/schemas.py`
- implement route handlers in `src/inference_service/routes.py`
- define batch input and output contracts in `src/inference/contracts.py` and `src/inference/batch_predict.py`
- keep `configs/service/default.yaml` as the canonical service config entry

### Success criteria

- one shared inference core serves single-image, batch, and API use cases
- API responses include label, confidence, `needs_review`, and provenance metadata
- batch and API predictions are numerically consistent on the same input
- the serving interface is documented and smoke-testable

---

## Phase F — Monitoring and observability

Goal: make model behavior observable in production and capable of drift detection.

### Current state

Repo reality today:

- `src/monitoring/prediction_logger.py` exists but is a placeholder
- `src/monitoring/drift.py` exists but is a placeholder
- `src/monitoring/reporting.py` exists in the module tree, but the monitoring layer is not production-ready

There is no implemented service metric exposure or drift report pipeline yet.

### Next concrete tasks

- define structured prediction logging in `src/monitoring/prediction_logger.py`
- implement simple statistical drift checks in `src/monitoring/drift.py`
- add report generation in `src/monitoring/reporting.py`
- integrate latency, error rate, and confidence summaries into the future API surface
- document alert conditions tied to confidence drift, latency drift, and review-rate spikes

### Success criteria

- predictions can be logged with timestamp, model version, confidence, and request metadata
- offline drift reports can compare a reference window vs a live window
- service metrics expose latency and error-rate visibility
- alert thresholds are defined for operational review

---

## Phase G — Lifecycle orchestration: registry, promotion, rollback

Goal: move from experiment tracking only to lifecycle orchestration with controlled promotion.

### Current state

Implemented today:

- MLflow tracking URI and experiment name are configured in `configs/mlflow/default.yaml`
- `src/training/train.py` logs parameters, metrics, and artifacts to MLflow
- `mlflow.db` exists in the repo workspace

Limitations today:

- `registry_uri` is `null` in config
- `src/mlops/tracking.py`, `registry.py`, and `promote_model.py` are placeholders
- there is no actual champion/challenger or rollback flow implemented

### Next concrete tasks

- implement registry access and version lookup in `src/mlops/registry.py`
- implement promotion policy tooling in `src/mlops/promote_model.py`
- emit evaluation-gate summary artifacts that promotion logic can consume
- define rollback semantics around incumbent model version and config provenance
- decide whether MLflow Model Registry or a simpler artifact manifest should be the first production target

### Success criteria

- candidate and champion models have explicit version identities
- promotion decisions are traceable to evaluation artifacts
- rollback can restore a known-good version and config set
- lifecycle orchestration is no longer manual or implicit

---

## Phase H — CI/CD for ML and Dockerized execution

Goal: support CI/CD for ML rather than relying on local-only experimentation.

### Current state

Repo reality today:

- `.github/workflows/ci.yaml` exists but is empty
- `docker/Dockerfile.api` and `docker/Dockerfile.train` exist but are empty
- tests are present in `tests/`
- current test coverage includes manifest building, splitters, preprocessing, validation, and datamodule behavior

### Next concrete tasks

- populate `.github/workflows/ci.yaml` with lint, unit-test, and smoke-eval jobs
- populate Dockerfiles for training and serving environments
- keep smoke tests lightweight and artifact-aware
- add CI checks for placeholder interfaces as they become implemented
- add a release workflow that runs evaluation gates before promotion steps

### Success criteria

- every pull request runs automated validation on core data and training/evaluation utilities
- Docker images can build reproducibly for train and serve targets
- CI/CD for ML supports candidate evaluation before merge or promotion
- basic smoke tests protect the serving interface from regressions

---

## Phase I — Performance optimization track

Goal: make latency, throughput, and cost first-class engineering constraints.

### Current state

Implemented today:

- model families include both ResNet50 and ViT configurations under `configs/model/`
- inference code can run on CPU or CUDA depending on availability

Not implemented today:

- no benchmark suite for latency or throughput
- no batch-vs-online performance comparison
- no ONNX / TorchScript export path confirmed in the repo

### Next concrete tasks

- add timing instrumentation to batch and API inference once those surfaces are implemented
- benchmark CPU-only vs GPU inference on representative images
- compare model families for cost/performance, not only accuracy
- document whether export formats such as ONNX or TorchScript are worth adding based on measured bottlenecks

### Success criteria

- benchmark artifacts report latency and throughput under defined hardware assumptions
- deployment recommendations distinguish interactive API from offline batch processing
- model selection includes quality/cost trade-offs
- optimization work is guided by measured bottlenecks rather than guesswork

---

## Phase J — Governance, privacy, and compliance posture

Goal: document a realistic compliance stance for a public-data demo and a path toward restricted deployments.

### Current state

Repo reality today:

- datasets are public and non-personal
- there is no implemented access control, retention policy, or audit trail layer
- current use case avoids PII by design

### Next concrete tasks

- keep the README explicit that this is a public-data demo
- add audit and retention requirements to future serving and monitoring docs
- ensure prediction metadata includes traceability fields once the service surface is implemented
- document how the stack would adapt to on-prem or restricted client environments

### Success criteria

- demo documentation is honest about privacy scope and implementation limits
- future production adaptation requirements are explicit
- governance is treated as part of system design, not an afterthought

---

## Operating model summary

Across all phases, the intended product behavior is:

1. version data deterministically
2. train reproducibly
3. evaluate against release gates
4. expose a stable inference contract
5. monitor drift and operational health
6. promote or roll back based on evidence
7. feed reviewed cases back into retraining

That is the production-first, inspection-triage interpretation of this repository.

---

## Near-term priority stack

If this roadmap is executed incrementally, the highest-value next steps are:

1. make `CrackDataModule` consume persisted manifest/split artifacts
2. standardize evaluation outputs for release-gate readiness
3. finish the shared inference core and stable API/batch schemas
4. wire monitoring and promotion scaffolding into real lifecycle tooling
5. turn empty CI and Docker placeholders into working ML delivery infrastructure

These steps most directly support the job-offer themes of **industrialisation**, **evaluation-first ML**, **deployable inference**, **monitoring/drift**, and **CI/CD for ML**.