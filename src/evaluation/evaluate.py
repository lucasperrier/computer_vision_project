import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    import mlflow
except Exception:
    mlflow = None

from src.data.datamodule import CrackDataModule
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_dict(cfg: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def build_model(cfg: Dict[str, Any]):
    model_name = str(cfg.get("model", "resnet50")).lower()
    if "vit" in model_name:
        return VisionTransformerModule(cfg)
    return ResNet50Module(cfg)


@torch.no_grad()
def predict_on_test(
    model: torch.nn.Module,
    datamodule: CrackDataModule,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      y_true: (N,)
      y_pred: (N,)
      y_prob_pos: (N,) probability of class 1
      avg_loss: float
    """
    model.eval()
    model.to(device)

    dl = datamodule.test_dataloader()
    ce = torch.nn.CrossEntropyLoss()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    losses = []

    for batch in dl:
        x, y = batch
        x = x.to(device)
        y = torch.as_tensor(y, device=device).long()

        logits = model(x)
        loss = ce(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())
        y_prob_all.append(probs[:, 1].detach().cpu().numpy())
        losses.append(loss.item())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_prob_pos = np.concatenate(y_prob_all, axis=0)

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return y_true, y_pred, y_prob_pos, avg_loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["test_acc"] = float(accuracy_score(y_true, y_pred))
    metrics["test_f1"] = float(f1_score(y_true, y_pred, average="binary"))

    # AUROC requires both classes present
    try:
        metrics["test_auc"] = float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        metrics["test_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    # Helpful engineering view
    tn, fp, fn, tp = cm.ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    # Derived rates
    metrics["precision_pos"] = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    metrics["recall_pos"] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    metrics["specificity_neg"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=["no_crack", "crack"], digits=4, zero_division=0
    )
    return metrics


def maybe_log_mlflow(cfg: Dict[str, Any], metrics: Dict[str, Any], artifacts_dir: Path):
    if not cfg.get("use_mlflow", False):
        return
    if mlflow is None:
        raise RuntimeError("use_mlflow=true but mlflow is not installed/importable.")

    tracking_uri = cfg.get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("experiment_name", "crack_detection")
    run_name = cfg.get("run_name", "eval_run")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_dict(cfg))

        # Log scalar metrics only
        for k, v in metrics.items():
            if isinstance(v, (float, int)) and v is not None:
                mlflow.log_metric(k, float(v))

        # Log artifacts (json, confusion matrix, predictions)
        for p in artifacts_dir.rglob("*"):
            if p.is_file():
                mlflow.log_artifact(str(p), artifact_path="eval_artifacts")


def main(config_path: str):
    cfg = load_config(config_path)

    if cfg.get("deterministic", False):
        seed_everything(int(cfg.get("seed", 42)), workers=True)

    output_dir = Path(cfg.get("output_dir", "reports/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    datamodule = CrackDataModule(
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 4)),
        val_split=float(cfg.get("val_split", 0.1)),
        test_split=float(cfg.get("test_split", 0.1)),
        robustness_split=float(cfg.get("robustness_split", 0.1)),
    )
    datamodule.setup(stage="test")

    # Model
    ckpt_path = cfg.get("checkpoint_path", None)
    model = build_model(cfg)

    # If checkpoint provided, load weights into the model
    if ckpt_path:
        ckpt_path = str(ckpt_path)
        model_cls = VisionTransformerModule if "vit" in str(cfg.get("model", "")).lower() else ResNet50Module
        # load_from_checkpoint will restore weights; pass cfg as init argument
        model = model_cls.load_from_checkpoint(ckpt_path, config=cfg)

    # Run Lightning test loop too (optional sanity) — uses model.test_step metrics
    # Note: this will only be correct if your datamodule labels are tensors/long.
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator="auto",
        deterministic=bool(cfg.get("deterministic", False)),
    )
    _ = trainer.test(model, datamodule=datamodule, verbose=True)

    # Manual evaluation to produce confusion matrix/report/pred dump
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred, y_prob_pos, avg_loss = predict_on_test(model, datamodule, device=device)

    metrics = compute_metrics(y_true, y_pred, y_prob_pos)
    metrics["test_loss"] = float(avg_loss)
    metrics["checkpoint_path"] = ckpt_path

    # Save metrics
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(metrics["classification_report"], encoding="utf-8")

    # Save confusion matrix
    if cfg.get("save_confusion_matrix", True):
        cm_path = output_dir / "confusion_matrix.npy"
        np.save(cm_path, np.array(metrics["confusion_matrix"], dtype=np.int64))

    # Save predictions
    if cfg.get("save_predictions", True):
        pred_path = output_dir / "predictions.npz"
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, y_prob_pos=y_prob_pos)

    # Print a compact summary
    print("\n=== Test metrics ===")
    for k in ["test_loss", "test_acc", "test_f1", "test_auc", "precision_pos", "recall_pos", "specificity_neg"]:
        print(f"{k}: {metrics.get(k)}")
    print("\nConfusion matrix [[tn, fp],[fn, tp]]:")
    print(np.array(metrics["confusion_matrix"], dtype=np.int64))
    print("\nClassification report:")
    print(metrics["classification_report"])

    # Optional MLflow logging
    maybe_log_mlflow(cfg, metrics, artifacts_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a crack detection model on the test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation YAML config")
    args = parser.parse_args()
    main(args.config)