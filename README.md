# SentinelInspect

## Production-First Visual Inspection Triage System

SentinelInspect reframes this repository from a pure crack-classification project into a **production-first visual inspection triage system** for quality control.

Today, the repo already contains a real backbone for:

- deterministic data inventory through manifest generation
- dataset validation and persisted split artifacts
- reproducible training jobs driven by Hydra configuration
- offline evaluation with saved reports and prediction artifacts
- single-image inference from trained checkpoints
- MLflow experiment tracking

It also contains **scaffolding / placeholders** for the rest of a production ML surface:

- batch inference CLI wrappers
- FastAPI serving layer
- monitoring and drift reporting
- model registry / promotion / rollback workflows
- Docker images and CI/CD for ML automation

The product goal is not “just classification”. It is a **decision-support workflow** that helps reduce manual inspection workload while controlling operational risk under noisy images, imperfect data quality, and domain shift.

---

## Why this matters in industry

In real quality-control and visual inspection workflows, a model is useful only if it can operate under uncertainty.

A production system must handle:

- noisy or compressed images from real acquisition pipelines
- variable lighting, blur, resolution, and background texture
- domain shift between collection sites or asset types
- asymmetric risk, where missed defects are more costly than extra reviews
- traceability requirements for model version, preprocessing version, and dataset provenance

That is why this repository is positioned around **industrialisation**, **evaluation-first ML**, and **deployable inference**, not only benchmark accuracy.

---

## Product workflow

This repo is now framed as the backbone of a visual inspection triage feature:

1. An inspector uploads or selects one or more inspection images.
2. The system returns a predicted label and confidence.
3. The system can route uncertain cases to `needs_review` / human review.
4. A reviewer confirms or overrides the result.
5. Confirmed feedback becomes a future retraining signal.

This product framing supports two operational concepts that matter in ML Engineer / MLOps roles:

- **operating point selection**: choosing thresholds that trade off review volume vs miss rate
- **human-in-the-loop triage**: using the model to prioritize attention instead of pretending full autonomy

---

## System behavior: triage contract

The intended output contract for each input image is:

- `predicted_label`: current task is effectively `crack` / `no_crack`
- `confidence_score`: class probability or score from the model
- `needs_review`: abstention / triage flag when confidence is low or risk is high
- `metadata`: provenance such as model version, preprocessing configuration, dataset/split lineage when available

### Current implementation status

Implemented today:

- predicted class output in `src/inference/predict.py`
- class probabilities in `src/inference/predict.py`
- checkpoint path and model name in inference output
- evaluation artifacts persisted under `reports/eval*/`

Target behavior, not fully implemented yet:

- `needs_review` / abstain flag
- calibrated confidence
- stable API response schema
- explicit dataset/split version embedded in prediction responses
- promotion-based model versioning for deployment

For portfolio and production-readiness purposes, the system should be read as a **triage system with abstention as a target capability**, even though abstention is still **in progress**.

---

## Production surface

SentinelInspect is structured around the production-first ML lifecycle expected in ML Engineer / MLOps roles:

- **data contract**: raw images -> manifest -> persisted split files -> validation gates
- **training job**: Hydra-configured training entrypoint with deterministic seeding and MLflow logging
- **evaluation suite**: offline evaluation with saved metrics, confusion matrices, reports, and prediction artifacts
- **serving target**: reusable inference core intended to power both API and batch workflows
- **monitoring target**: prediction logging, drift checks, and operational metrics
- **promotion target**: candidate evaluation before promote / rollback decisions

Text diagram:

`Data -> Train -> Evaluate (gates) -> Register/Promote -> Serve -> Monitor -> Retrain`

### What is production-grade today vs planned

Production-oriented and implemented today:

- manifest builder: `src/data/build_manifest.py`
- split generator: `src/data/splitters.py`
- dataset validator: `src/data/validate_dataset.py`
- reproducible config loading: `src/config/load.py`, `src/config/schema.py`
- training job: `src/training/train.py`
- offline evaluation job: `src/evaluation/evaluate.py`
- shared transforms module: `src/preprocessing/transforms.py`
- single-image inference path: `src/inference/predict.py`
- experiment tracking with MLflow via `configs/mlflow/default.yaml`

Present but still scaffolding / placeholder:

- `scripts/*.py` command wrappers
- `src/inference/batch_predict.py`
- `src/inference_service/app.py`, `routes.py`, `schemas.py`
- `src/monitoring/*.py`
- `src/mlops/*.py`
- `src/jobs/*.py`
- `.github/workflows/ci.yaml`
- `docker/Dockerfile.api`, `docker/Dockerfile.train`

---

## One-command workflows

The repository currently has **real runnable modules under `src/`** and **placeholder wrappers under `scripts/`**. The commands below reflect what is actually implemented today.

### Current commands (implemented today)

#### Data preparation

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

### Intended interface (present as placeholders, not stable yet)

The following files exist but are currently placeholders or empty wrappers:

- `scripts/build_manifest.py`
- `scripts/validate_dataset.py`
- `scripts/train_model.py`
- `scripts/evaluate_model.py`
- `scripts/batch_predict.py`
- `src/inference/batch_predict.py`
- `src/inference_service/app.py`

So for now, treat the `src.*` module entrypoints as the real interface and the `scripts/` surface as **in progress**.

---

## Evaluation-first: benchmarks and release gates

This repo follows an **evaluation-first** direction aligned with production ML roles: models should not be promoted based only on one headline metric.

### Current evaluation artifacts

Implemented today:

- test-set evaluation in `src/evaluation/evaluate.py`
- metrics: accuracy, F1, ROC-AUC when available
- confusion matrix export
- text classification report export
- prediction arrays export for downstream analysis
- explainability utilities under `src/explainability/`
- localization/faithfulness helper metrics in `src/evaluation/robustness.py`
- historical evaluation outputs already stored in `reports/eval_pretrained_*` and `reports/eval_trained_*`

### Current limitations

Not yet fully implemented:

- automated regression comparison against a previous champion model
- calibration metrics such as ECE or Brier score
- selective prediction / abstention curves
- formal OOD evaluation job
- corruption benchmark suite with severity sweeps
- release gate enforcement tied to promotion

### Release-gate policy target

The intended release logic is:

1. Train a candidate model.
2. Run standardized offline evaluation.
3. Compare against the current reference model.
4. Check pass/fail gates on quality, robustness, and operational constraints.
5. Promote only if gates pass; otherwise keep the incumbent and investigate.

Target gate dimensions:

- core classification quality
- robustness to blur, compression, noise, brightness shifts
- domain-shift / OOD sensitivity
- calibration quality
- selective prediction behavior under review thresholds
- artifact completeness and provenance logging

This maps directly to the “prototype -> production performance” and “benchmark / regression / release gate” expectations in ML Engineer and MLOps job descriptions.

---

## Deployment and serving

### Shared inference core

The repo already contains a reusable inference module in `src/inference/predict.py`. The intended architecture is to reuse the same model-loading and preprocessing behavior across:

- single-image inference
- batch inference
- API serving

This is important to reduce training/serving skew.

### Current serving status

Implemented today:

- single-image checkpoint-based inference
- service config placeholder in `configs/service/default.yaml`
- shell entrypoint placeholder: `scripts/start_api.sh`

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