# SentinelInspect

## Production-First Visual Inspection Triage System

SentinelInspect reframes crack classification into a **production-first visual inspection triage system** for quality control: reduce manual review volume while controlling miss risk under noise and domain shift.

**Implemented today (real entrypoints in `src/`):**

- data inventory: `src/data/build_manifest.py`
- deterministic splits: `src/data/splitters.py`
- dataset validation: `src/data/validate_dataset.py`
- Hydra + Lightning training job: `src/training/train.py`
- offline eval artifacts: `src/evaluation/evaluate.py`
- single-image inference from checkpoint: `src/inference/predict.py`
- MLflow tracking: `configs/mlflow/default.yaml`

**Planned / scaffolded (present but not finished):** batch inference, FastAPI service, monitoring/drift, registry/promotion, CI + Docker.

---

## What “triage” means here

The goal isn’t “autonomous inspection”. It’s **decision support**:

1. input image(s)
2. model returns label + confidence
3. low-confidence cases are routed to review (`needs_review` is a target capability)
4. reviewer feedback becomes retraining signal

---

## Output contract (target)

Per image:

- `predicted_label` (currently effectively `crack` / `no_crack`)
- `confidence_score` (implemented in `src/inference/predict.py`)
- `needs_review` (planned: thresholded abstention)
- `metadata` (planned: model + preprocessing + data lineage)

---

## Evaluation-first release posture

This repo is structured to support “**train → evaluate → promote**” instead of shipping by accuracy alone.

**Artifacts already produced by `src/evaluation/evaluate.py`:** `metrics.json`, `classification_report.txt`, `confusion_matrix.npy`, `predictions.npz` under `reports/eval*/`.

**Still to add (planned):** candidate-vs-champion regression checks, calibration (ECE/Brier), selective prediction curves, OOD + corruption benchmarks, and CI-enforced gates.

---

## What exists vs what’s scaffolded

- **Use `python -m src...` entrypoints** (implemented).
- Treat `scripts/`, `src/inference_service/`, `src/monitoring/`, `src/mlops/`, `src/jobs/`, `.github/workflows/`, and `docker/` as **in progress** placeholders.

---

## Quickstart (implemented modules)

```bash
python -m src.data.build_manifest --raw-root data/raw --output-csv data/processed/manifests/manifest.csv --output-json data/processed/manifests/manifest.json
python -m src.data.splitters --manifest-path data/processed/manifests/manifest.csv --output-dir data/processed/splits
python -m src.data.validate_dataset --manifest data/processed/manifests/manifest.csv --train data/processed/splits/train.csv --val data/processed/splits/val.csv --test data/processed/splits/test.csv --robustness data/processed/splits/robustness.csv --raw-root .
```

#### Train

```bash
python -m src.training.train
```

Hydra overrides are supported because training is configured through `configs/train.yaml` and related config groups.

#### Evaluate

```bash
python -m src.evaluation.evaluate
```

This writes artifacts such as:

- `reports/eval/metrics.json`
- `reports/eval/classification_report.txt`
- `reports/eval/confusion_matrix.npy`
- `reports/eval/predictions.npz`

#### Single-image inference

```bash
python -m src.inference.predict image_path=/absolute/path/to/image.jpg checkpoint_path=/absolute/path/to/model.ckpt
```


## Serving (current vs target)

- **Current:** single-image checkpoint inference via `src/inference/predict.py`.
- **Target:** shared inference core powering batch + FastAPI, with stable schemas and monitoring hooks.

Scaffolding / placeholder:

- FastAPI app layer in `src/inference_service/`
- stable request/response schemas
- batch inference service surface
- registry-backed loading for deployed models

### Intended API contract

The intended API response shape for production serving is:

- input reference or request id
- predicted label
- confidence score
- `needs_review`
- model metadata / provenance
- optional explanations or trace ids

That response contract is a **product target**, not a fully wired implementation yet.

---

## Performance and constraints

This section uses **measured facts only where they exist** and labels the rest as targets.

### Latency and throughput

Measured in repo today:

- no formal latency benchmark artifacts found
- no throughput benchmark artifacts found

Targets for the production-first triage direction:

- online API path: low-latency single-image scoring suitable for decision support
- batch path: throughput-optimized offline scoring for larger inspection queues
- separate operating modes for CPU-only deployment vs GPU-backed deployment

Status: **not yet benchmarked in-repo**.

### Cost awareness

Current repo reality:

- training supports both ResNet and ViT model families
- inference code can run on CPU or CUDA depending on availability
- no cost benchmark or autoscaling policy is implemented yet

Target production guidance:

- use batch workflows where latency is not critical
- reserve online scoring for interactive inspection workflows
- choose CPU vs GPU deployment based on SLA and queue volume

### Security posture

Current demo posture:

- datasets under `data/raw/` are public research datasets
- no PII handling is implemented or required by the current use case

Adaptation path for restricted client environments:

- deploy in on-prem or controlled VPC environments
- add access control and authentication to serving endpoints
- enforce artifact retention and audit logging policies
- separate model artifacts, data artifacts, and prediction logs by environment

Status: **security hardening for real production is planned, not implemented**.

---

## Monitoring and drift

Monitoring is a required part of the production-first framing, especially for visual inspection where drift can silently degrade quality.

### Monitoring requirements for this system

The operating system should eventually track:

- confidence drift
- input distribution drift
- data quality drift
- latency and error-rate monitoring
- review-rate changes (`needs_review` volume)
- class prevalence changes over time

### Repo status today

- `src/monitoring/prediction_logger.py` exists but is a placeholder
- `src/monitoring/drift.py` exists but is a placeholder
- `src/monitoring/reporting.py` exists in the tree, but the monitoring surface is not yet documented as production-ready

So observability should be read as **planned capability / scaffolding**, not as a finished monitoring stack.

Target deliverables on this track:

- structured prediction logs
- Prometheus-style service metrics
- offline drift reports
- alert thresholds on confidence, latency, and review-rate changes

---

## Privacy and compliance posture (demo)

This demo intentionally avoids personal data.

- current datasets are public and non-personal
- no user identity, biometric, or customer-sensitive content is part of the demonstrated workflow
- the repo therefore avoids many RGPD / privacy constraints by design in its current form

For a real industrial deployment, the system would still need:

- audit trails for predictions and overrides
- environment-level access control
- retention and deletion policies
- separation of duties across labeling, review, and deployment
- traceability of model version, data version, and threshold configuration

These controls are described here as **deployment posture expectations**, not claimed as implemented features.

---

## MLOps and lifecycle orchestration

This repository already demonstrates the start of an MLOps-oriented shape.

Implemented today:

- MLflow experiment tracking with local SQLite backend configured in `configs/mlflow/default.yaml`
- training job logs params, checkpoints, and metrics to MLflow
- saved evaluation artifacts under `reports/`

Scaffolding / planned:

- model registry integration
- champion/challenger promotion workflow
- rollback workflow
- automated offline evaluation jobs in `src/jobs/`
- CI/CD for ML checks in `.github/workflows/ci.yaml`

This maps directly to job-offer expectations around **CI/CD for ML**, **lifecycle orchestration**, and **prototype-to-production performance**.

---

## Project status

### Implemented

- manifest generation from raw image folders in `src/data/build_manifest.py`
- deterministic split generation in `src/data/splitters.py`
- dataset validation and split-overlap checks in `src/data/validate_dataset.py`
- shared preprocessing under `src/preprocessing/`
- Hydra-configured training job in `src/training/train.py`
- MLflow experiment tracking during training
- offline evaluation with persisted reports in `src/evaluation/evaluate.py`
- explainability modules under `src/explainability/`
- single-image inference from checkpoint in `src/inference/predict.py`
- unit tests for manifest building, splitters, validation, preprocessing, and datamodule-related behavior

### In progress / planned

- persisted manifest-and-splits as the single source of truth for training
- abstention / `needs_review` behavior
- calibration and selective prediction
- standardized OOD and corruption benchmarking
- batch inference interface
- FastAPI serving API
- monitoring, drift, and alerting
- model registry, promotion, and rollback
- CI/CD for ML workflow automation
- Dockerized training and API runtime
- latency / throughput benchmarking

---

## Repository structure

This structure matches what is actually present in the repository today.

```text
.
├── configs/
│   ├── data/               # Data paths and split ratios
│   ├── mlflow/             # Tracking configuration
│   ├── model/              # ResNet50 and ViT config groups
│   ├── service/            # Service host/port placeholder config
│   ├── trainer/            # Lightning trainer config
│   ├── eval.yaml           # Hydra evaluation entry config
│   ├── inference.yaml      # Hydra inference entry config
│   └── train.yaml          # Hydra training entry config
├── data/
│   ├── raw/                # Public crack datasets and archives
│   ├── interim/
│   └── processed/
│       ├── features/
│       ├── manifests/
│       └── splits/
├── docker/                 # Present, but Dockerfiles are placeholders
├── docs/
│   ├── archiecture.md      # Placeholder
│   ├── roadmap.md          # Production-first capability roadmap
│   └── serving_design.md   # Placeholder
├── reports/
│   ├── eval*/              # Saved evaluation reports and prediction artifacts
│   └── ...
├── runs/                   # Saved checkpoints / run outputs
├── scripts/                # Placeholder wrappers / shell helpers
├── src/
│   ├── config/             # Typed config loading and schema validation
│   ├── data/               # Manifest generation, splitters, validation, datamodule
│   ├── evaluation/         # Offline evaluation and robustness helpers
│   ├── explainability/     # Grad-CAM, SHAP, explainability utilities
│   ├── inference/          # Implemented single-image inference; batch placeholder
│   ├── inference_service/  # FastAPI scaffolding / placeholders
│   ├── jobs/               # Placeholder job modules for future orchestration
│   ├── mlops/              # ML lifecycle scaffolding / placeholders
│   ├── models/             # ResNet50, ViT, factory logic
│   ├── monitoring/         # Monitoring scaffolding / placeholders
│   ├── preprocessing/      # Shared transforms and preprocessing
│   ├── training/           # Training job and reproducibility helpers
│   └── utils/
├── tests/
│   ├── fixtures/
│   ├── integration/
│   ├── unit/
│   └── test_datamodule.py
├── README.md
├── requirements.txt
└── mlflow.db
```

---

## Documentation reality check

Some docs referenced in older versions of the README are not yet written or are empty placeholders. At the time of this rewrite:

- `docs/roadmap.md` is maintained and updated
- `docs/archiecture.md` exists but is currently a placeholder
- `docs/serving_design.md` exists but is currently a placeholder

---

## Author

**Lucas Perrier**  
MSc in Data & Artificial Intelligence, ESILV  
Interests: scientific machine learning, uncertainty-aware modelling, spatio-temporal systems, and robust applied ML

GitHub: [github.com/lucasperrier](https://github.com/lucasperrier)