
# Roadmap

## SentinelInspect roadmap

### Ship the strongest version of what this repository already is

This roadmap is written against the **current state of this repo**, not an idealized platform.

The goal is to turn the existing crack-classification project into a **production-first visual inspection triage system** that demonstrates four concrete capabilities:

1. reproducible data inputs
2. standardized evaluation outputs
3. deployable inference
4. basic delivery infrastructure

The governing product question is:

> Can this repository show that a computer vision prototype was turned into a controlled, testable, deployable inspection workflow instead of stopping at training and offline metrics?

That is the hiring signal this roadmap is optimizing for.

---

## Product framing

SentinelInspect is a **visual inspection triage system** for crack classification.

It is not positioned as full autonomous inspection.

It is positioned as a practical workflow that can:

1. ingest an image
2. predict `crack` or `no_crack`
3. return a confidence score
4. mark low-confidence cases for review
5. save outputs for later analysis

That framing matches what the repository can credibly support with a focused v1.

---

## What is already real in this repo

The repository already contains meaningful implementation work:

### Data foundations

- `src/data/build_manifest.py` builds a manifest from `data/raw/`
- `src/data/splitters.py` generates deterministic split files from the manifest
- `src/data/validate_dataset.py` validates duplicates, missing files, corrupt images, required columns, and split overlap
- processed artifact directories already exist under:
	- `data/processed/manifests/`
	- `data/processed/splits/`

### Training and experiment flow

- Hydra config structure is in place under `configs/`
- training and evaluation entrypoints exist through:
	- `configs/train.yaml`
	- `configs/eval.yaml`
	- `configs/inference.yaml`
- MLflow configuration exists in `configs/mlflow/default.yaml`
- model code exists under `src/models/`
- training code exists under `src/training/`
- saved checkpoints already exist under `runs/`

### Evaluation and reporting

- `src/evaluation/evaluate.py` already computes test metrics and writes artifacts
- evaluation outputs already exist in `reports/`, including:
	- `metrics.json`
	- `classification_report.txt`
	- `confusion_matrix.npy`
	- `predictions.npz`
- historical evaluation folders already exist for pretrained and trained models

### Inference and explainability

- `src/inference/predict.py` performs single-image checkpoint inference
- explainability modules already exist under `src/explainability/`
- configs for service and inference already exist under `configs/service/` and `configs/inference.yaml`

### Repo structure and delivery scaffolding

- `requirements.txt` already lists the core Python dependencies
- `docker/` exists with `Dockerfile.api` and `Dockerfile.train`
- `.github/workflows/ci.yaml` exists
- tests already exist in `tests/`, including at least `tests/test_datamodule.py` plus `tests/unit/` and `tests/integration/`

This means the repo is **not** starting from zero. The strongest next step is to finish the highest-signal surfaces cleanly.

---

## What is partial or unfinished today

The main gaps are also visible in the repo:

### Data loading gap

- `src/data/datamodule.py` still builds datasets by scanning raw folders and doing runtime splitting with `train_test_split`
- that means persisted manifest and split artifacts are **not yet the canonical source of truth** for training/evaluation

### Evaluation standardization gap

- `src/evaluation/evaluate.py` already contains working metric logic directly in the file
- `src/evaluation/metrics.py` still exists as an unfinished placeholder surface
- there is no clean confidence-based triage analysis or explicit `needs_review` release rule yet

### Inference productization gap

- `src/inference/predict.py` works for single-image inference
- `src/inference/batch_predict.py` still needs to be finished
- `src/inference_service/` exists with `app.py`, `routes.py`, `schemas.py`, `dependencies.py`, and `logging.py`, but the service layer is still an incomplete scaffold
- there is not yet one shared inference core used consistently by CLI, batch, and API

### Delivery gap

- `pyproject.toml` exists but is currently empty
- `.github/workflows/ci.yaml` exists but still needs to become a real CI workflow
- `docker/Dockerfile.api` and `docker/Dockerfile.train` exist but still need real build instructions
- some wrappers in `scripts/` exist, but the highest-value core interfaces should be stabilized first

### Monitoring and MLOps gap

- `src/monitoring/` exists with `prediction_logger.py`, `drift.py`, and `reporting.py`
- `src/mlops/` exists with `artifact_store.py`, `promote_model.py`, `registry.py`, and `tracking.py`
- these directories are useful future-facing structure, but they should remain **small follow-on work**, not the main deliverable for v1

---

## Scope principle

This repo becomes stronger by finishing a **small number of visible engineering surfaces well**.

It does not become stronger by pretending to be a full production platform before the core workflow is complete.

So the scope should stay divided into:

- **must ship**
- **nice to have**
- **defer**

---

## Must ship

### Phase 1 — Make persisted data artifacts the source of truth

#### Goal

Make the saved manifest and split files the canonical dataset contract.

#### Repo-specific reason

You already have the right artifact builders:

- `src/data/build_manifest.py`
- `src/data/splitters.py`
- `src/data/validate_dataset.py`

The missing step is wiring that contract into the runtime path used by training and evaluation.

Right now, `src/data/datamodule.py` still scans:

- `data/raw/sdnet2018`
- `data/raw/ccic`

and creates splits dynamically with `train_test_split`.

That weakens reproducibility because the persisted split artifacts are not yet the thing the rest of the system actually consumes.

#### Tasks

- refactor `src/data/datamodule.py` to load from:
	- `data/processed/manifests/manifest.csv`
	- `data/processed/splits/train.csv`
	- `data/processed/splits/val.csv`
	- `data/processed/splits/test.csv`
- decide whether `robustness.csv` remains a true evaluation split or a later extension
- route all dataset paths through config instead of hard-coded folder assumptions
- ensure validation from `src/data/validate_dataset.py` can block downstream training/evaluation if artifacts are invalid

#### Success criteria

- training no longer performs hidden runtime re-splitting
- evaluation uses the same saved split artifacts as training
- manifest and split CSVs are enough to reconstruct the dataset membership used in an experiment
- invalid manifests or overlapping splits fail fast

---

### Phase 2 — Standardize the evaluation bundle around what `src/evaluation/evaluate.py` already writes

#### Goal

Turn evaluation into the main release artifact of the project.

#### Repo-specific reason

You already have a working evaluation path.

`src/evaluation/evaluate.py` currently:

- loads a model
- runs test inference
- computes:
	- accuracy
	- F1
	- ROC AUC when possible
	- confusion matrix
	- classification report
- writes:
	- `metrics.json`
	- `classification_report.txt`
	- `confusion_matrix.npy`
	- `predictions.npz`

So the problem is not “evaluation does not exist.”

The real problem is that the evaluation contract is not yet fully standardized or confidence-aware.

#### Tasks

- move reusable metric helpers into `src/evaluation/metrics.py`
- keep `src/evaluation/evaluate.py` as the main entrypoint, but make its outputs a stable bundle
- extend the saved outputs so they clearly include:
	- core classification metrics
	- confusion matrix
	- classification report
	- per-sample predictions
	- positive-class confidence scores
	- a simple threshold analysis for manual review routing
- define a lightweight first-pass triage rule such as:
	- mark predictions as `needs_review` when confidence is below a chosen threshold band

#### Success criteria

- every evaluated model produces the same artifact structure in `reports/`
- confidence scores are saved in a reusable form for downstream analysis
- `needs_review` can be derived directly from saved prediction outputs
- evaluation outputs can support a simple deployment decision instead of being just historical logs

---

### Phase 3 — Finish shared inference and expose a usable API surface

#### Goal

Turn the repository into something that can actually serve predictions outside the training script.

#### Repo-specific reason

There is already a working single-image inference path in `src/inference/predict.py`.

That file already:

- loads config with Hydra
- loads a checkpoint
- preprocesses a single image
- runs the model
- returns class probabilities and predicted class

So the shortest path is to **extract and reuse the existing logic**, not rebuild inference from scratch.

#### Tasks

- extract a shared prediction core from `src/inference/predict.py`
- finish `src/inference/batch_predict.py`
- complete the service layer in:
	- `src/inference_service/schemas.py`
	- `src/inference_service/routes.py`
	- `src/inference_service/app.py`
- keep request/response contracts simple and explicit
- return at least:
	- predicted label
	- confidence
	- `needs_review`
	- model name / checkpoint / version metadata

#### Success criteria

- single-image, batch, and API inference all use the same prediction logic
- API and batch outputs are numerically consistent on the same image
- the service can be started locally and smoke-tested
- schemas make the inference interface easy to understand from the repo alone

---

### Phase 4 — Make packaging, CI, and Docker real

#### Goal

Close the gap between “good local project” and “serious ML engineering repo.”

#### Repo-specific reason

This is one of the clearest unfinished parts of the current repo:

- `pyproject.toml` is empty
- `.github/workflows/ci.yaml` exists but is not yet the real CI story
- `docker/Dockerfile.api` and `docker/Dockerfile.train` exist but are not yet implemented

Because these files are already visible at the top level, finishing them has outsized value for hiring signal.

#### Tasks

- populate `pyproject.toml` with project metadata and dependencies, likely aligned with `requirements.txt`
- turn `.github/workflows/ci.yaml` into a working GitHub Actions workflow
- include at minimum:
	- linting
	- unit tests
	- a lightweight smoke test for one core interface
- implement `docker/Dockerfile.train`
- implement `docker/Dockerfile.api`

#### Success criteria

- the repository can be installed reproducibly
- pull requests run automated checks
- train and API images build successfully
- core code paths are protected by CI instead of manual trust

---

## Nice to have

### Phase 5 — Add minimal monitoring hooks

This should stay intentionally small.

The monitoring layer should support the triage story, not become a full observability platform.

#### Repo-specific reason

You already have the module structure:

- `src/monitoring/prediction_logger.py`
- `src/monitoring/drift.py`
- `src/monitoring/reporting.py`

That is enough to add a lightweight but credible signal.

#### Tasks

- implement structured prediction logging in `src/monitoring/prediction_logger.py`
- log the minimum useful fields:
	- timestamp
	- model identifier
	- predicted label
	- confidence
	- `needs_review`
- add a very small offline report in `src/monitoring/reporting.py`
- keep drift analysis in `src/monitoring/drift.py` lightweight and honest

#### Success criteria

- inference outputs can be logged consistently
- one offline summary can compare reference vs recent confidence distributions
- the repo demonstrates post-deployment thinking without overstating platform maturity

---

### Phase 6 — Add small robustness and selective prediction support

This should remain tied to the review-routing story.

#### Repo-specific reason

You already have:

- `src/evaluation/robustness.py`
- `src/explainability/`
- saved historical evaluation folders under `reports/`

So a small extension here is plausible, but it should not displace the must-ship items.

#### Tasks

- extend evaluation artifacts with a few simple robustness checks
- add lightweight confidence and threshold summaries
- if feasible, add a small coverage-vs-risk style summary

#### Success criteria

- the repo can show how confidence and performance move under mild perturbations
- threshold selection for `needs_review` has some quantitative backing
- robustness remains supportive evidence, not a research detour

---

## Defer

The following should remain out of core scope unless the must-ship phases are already cleanly finished:

- large registry and promotion workflows in `src/mlops/`
- rollback automation
- complex champion/challenger orchestration
- extensive OOD benchmarking
- active learning loops
- pseudo-labeling
- distillation
- cloud deployment buildout
- Kubernetes
- ONNX or TorchScript export unless latency profiling proves it matters

These are fine future directions, but they should not compete with the strongest v1 story.

---

## Final target state for this repo

The shipped version of this repository should support the following concrete story:

### Data

`src/data/build_manifest.py`, `src/data/splitters.py`, and `src/data/validate_dataset.py` produce the dataset contract, and `src/data/datamodule.py` consumes it.

### Training

Training is config-driven through `configs/` and tracked in MLflow.

### Evaluation

`src/evaluation/evaluate.py` produces a standardized bundle in `reports/` with confidence-aware triage outputs.

### Inference

The same prediction core powers `src/inference/predict.py`, `src/inference/batch_predict.py`, and the FastAPI service under `src/inference_service/`.

### Delivery

The repo becomes installable, CI-checked, and Dockerized through `pyproject.toml`, `.github/workflows/ci.yaml`, `docker/Dockerfile.train`, and `docker/Dockerfile.api`.

### Monitoring

Predictions can be logged and summarized through the lightweight modules already present in `src/monitoring/`.

That is enough to make the project feel like a deployable ML engineering system rather than a training-only experiment.

---

## Recommended implementation order

Keep the implementation order strict:

1. wire manifest and split artifacts into `src/data/datamodule.py`
2. standardize the evaluation bundle in `src/evaluation/`
3. extract shared inference and finish `src/inference_service/`
4. complete `pyproject.toml`, CI, and Docker
5. add minimal monitoring hooks

If those five things are finished well, the repository becomes much stronger for hiring than a broader but less finished roadmap.

---

## Hiring signal this roadmap is optimized for

This repo should be able to support claims like:

- built a production-first computer vision triage system rather than a notebook-only classifier
- implemented deterministic dataset lineage through manifests and persisted split artifacts
- standardized evaluation outputs and added confidence-aware triage analysis
- shipped shared inference across single-image, batch, and API workflows
- added reproducible packaging, CI, and Dockerized execution

That is the level of specificity and scope this repository can credibly own today.

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