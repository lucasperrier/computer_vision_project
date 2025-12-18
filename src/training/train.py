import argparse
import shutil
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import CrackDataModule
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def flatten_dict(cfg, parent_key='', sep='.'):
    items = []
    for k, v in cfg.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def build_model(cfg):
    if 'vit' in cfg.get('model', '').lower():
        return VisionTransformerModule(cfg)
    return ResNet50Module(cfg)


def main(config_path: str):
    config = load_config(config_path)
    if config.get('deterministic', False):
        seed_everything(42, workers=True)

    tracking_uri = config.get('tracking_uri', 'file:./mlruns')
    experiment_name = config.get('experiment_name', 'crack_detection')
    run_name = config.get('run_name', config.get('model', 'run'))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_dict(config))
        mlflow.log_artifact(config_path, artifact_path='configs')

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            every_n_epochs=1,
            dirpath=f"runs/{experiment_name}-{run_name}",
            filename=f"{config.get('model', 'model')}-" + "{epoch:02d}-{val_loss:.4f}"
        )
        callbacks = [
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch')
        ]

        logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_id=run.info.run_id
        )

        datamodule = CrackDataModule(
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            val_split=config.get('val_split', 0.1),
            test_split=config.get('test_split', 0.1),
            robustness_split=config.get('robustness_split', 0.1)
        )

        model = build_model(config)

        trainer = Trainer(
            max_epochs=config.get('epochs', 30),
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=1,
            deterministic=config.get('deterministic', False),
            accelerator='auto',
            val_check_interval=float(config.get("val_check_interval", 1.0)),
        )

        trainer.fit(model, datamodule=datamodule)

        best_ckpt_path = checkpoint_callback.best_model_path
        exported_ckpt_path = None
        if best_ckpt_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = Path('trained_models') / experiment_name / timestamp
            export_dir.mkdir(parents=True, exist_ok=True)
            exported_ckpt_path = export_dir / Path(best_ckpt_path).name
            shutil.copy2(best_ckpt_path, exported_ckpt_path)
            mlflow.log_artifact(str(exported_ckpt_path), artifact_path='trained_models')
            mlflow.log_param('exported_checkpoint', str(exported_ckpt_path))

        val_metrics = trainer.validate(datamodule=datamodule, ckpt_path=best_ckpt_path or 'best')
        test_metrics = trainer.test(datamodule=datamodule, ckpt_path=best_ckpt_path or 'best')

        for metric_dict in val_metrics + test_metrics:
            for key, value in metric_dict.items():
                mlflow.log_metric(key, float(value))

        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            mlflow.log_metric(key, float(value))

        if exported_ckpt_path:
            print(f"Best checkpoint saved to: {exported_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crack detection models with MLflow tracking.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)