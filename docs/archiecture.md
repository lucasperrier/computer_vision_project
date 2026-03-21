# Current-state repository diagram

This diagram captures the **current state of the repository** as it exists today.

It is intentionally not an idealized target architecture.
It shows:

- the parts that are already implemented and used
- the parts that exist as scaffolding or partial surfaces
- the main execution flow from data to training, evaluation, and early inference

You can use this figure in the technical write-up section about the repo's transition from a research prototype toward a production-minded ML system.

## Current repo state

```mermaid
flowchart TD
	subgraph DataLayer[Data layer]
		RAW[data/raw\nSDNET2018 + CCIC]
		PROC[data/processed\nmanifests / splits / features]
		BUILD[src/data/build_manifest.py]
		SPLIT[src/data/splitters.py]
		VALIDATE[src/data/validate_dataset.py]
		DATAMODULE[src/data/datamodule.py\ncurrent runtime data loader]
	end

	subgraph ConfigLayer[Configuration layer]
		HYDRA[configs/\ntrain.yaml / eval.yaml / inference.yaml]
		CFG_DATA[configs/data/default.yaml]
		CFG_MODEL[configs/model/\nresnet50.yaml / vit.yaml]
		CFG_TRAINER[configs/trainer/default.yaml]
		CFG_SERVICE[configs/service/default.yaml]
		CFG_MLFLOW[configs/mlflow/default.yaml]
	end

	subgraph TrainingEval[Training and evaluation]
		TRAIN_SCRIPT[scripts/train_model.py]
		TRAIN[src/training/]
		MODELS[src/models/\nResNet / ViT]
		RUNS[runs/\ncheckpoints]
		MLFLOW[mlflow.db + mlruns/]
		EVAL_SCRIPT[scripts/evaluate_model.py]
		EVAL[src/evaluation/evaluate.py]
		REPORTS[reports/\nmetrics.json\nclassification_report.txt\nconfusion_matrix.npy\npredictions.npz]
		ROBUSTNESS[src/evaluation/robustness.py\npartial depth]
	end

	subgraph InferenceServing[Inference and serving]
		PREDICT[src/inference/\nsingle-image inference]
		BATCH[scripts/batch_predict.py\nplaceholder / incomplete]
		SERVICE[src/inference_service/\nAPI scaffold]
		DOCKER[docker/\nDockerfile.api / Dockerfile.train\nplaceholders]
	end

	subgraph Explainability[Explainability]
		XAI[src/explainability/\nGrad-CAM / SHAP / utilities]
	end

	subgraph QualityOps[Testing and ops surfaces]
		TESTS[tests/\nunit / integration / datamodule]
		MON[src/monitoring/\nplaceholder surfaces]
		MLOPS[src/mlops/\nregistry / tracking / promotion\nmostly scaffolded]
		CI[pyproject.toml + CI workflow\ncurrently incomplete]
	end

	RAW --> BUILD
	BUILD --> PROC
	PROC --> SPLIT
	PROC --> VALIDATE

	RAW -.current behavior: still rescanned directly.-> DATAMODULE
	PROC -.intended source of truth, not fully wired yet.-> DATAMODULE

	HYDRA --> TRAIN_SCRIPT
	CFG_DATA --> TRAIN_SCRIPT
	CFG_MODEL --> TRAIN_SCRIPT
	CFG_TRAINER --> TRAIN_SCRIPT
	CFG_MLFLOW --> TRAIN_SCRIPT

	TRAIN_SCRIPT --> TRAIN
	DATAMODULE --> TRAIN
	MODELS --> TRAIN
	TRAIN --> RUNS
	TRAIN --> MLFLOW

	HYDRA --> EVAL_SCRIPT
	EVAL_SCRIPT --> EVAL
	RUNS --> EVAL
	DATAMODULE --> EVAL
	EVAL --> REPORTS
	EVAL --> ROBUSTNESS
	EVAL --> XAI

	HYDRA --> PREDICT
	CFG_SERVICE --> SERVICE
	RUNS --> PREDICT
	PREDICT --> SERVICE
	PREDICT --> BATCH
	SERVICE --> DOCKER

	TESTS -.limited coverage around current implementation.-> DATAMODULE
	TESTS -.limited validation of system contracts.-> EVAL
	MON -.planned operational surface.-> SERVICE
	MLOPS -.planned lifecycle surface.-> RUNS
	CI -.not yet enforcing build/test path.-> TESTS

	classDef implemented fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px;
	classDef partial fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px;
	classDef placeholder fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1.5px;

	class BUILD,SPLIT,VALIDATE,DATAMODULE,HYDRA,CFG_DATA,CFG_MODEL,CFG_TRAINER,CFG_SERVICE,CFG_MLFLOW,TRAIN_SCRIPT,TRAIN,MODELS,RUNS,MLFLOW,EVAL_SCRIPT,EVAL,REPORTS,PREDICT,XAI,TESTS implemented;
	class PROC,ROBUSTNESS,BATCH,SERVICE,DOCKER,MON,MLOPS,CI partial;
```

## How to read it

- **Green** nodes represent parts that are already meaningfully implemented.
- **Amber** nodes represent partial, uneven, or only partly integrated surfaces.
- The dashed connection into `src/data/datamodule.py` highlights the main current-state issue:
  the repository already contains persisted manifests and splits, but the runtime training path still relies heavily on rescanning raw folders and doing split logic dynamically.

## Suggested caption for the report

**Figure — Current repository state.** The project already contains real components for data preparation, Hydra-based training, offline evaluation, MLflow tracking, and single-image inference, but several production-facing surfaces remain partial. In particular, persisted dataset artifacts, service boundaries, CI, packaging, and monitoring exist in the repository structure without yet being fully wired into the main execution path.

## Short LaTeX-ready summary

The diagram shows a repository that is beyond a notebook-only prototype but not yet a fully operational ML system. The most mature path runs from raw data through training, checkpointing, and offline evaluation. Around that path, the repo already contains the beginnings of production-minded interfaces --- manifests, deterministic splits, service scaffolding, Docker files, monitoring modules, and MLOps directories --- but many of those surfaces are still incomplete or only partially integrated.
