import argparse
import json
from pathlib import Path

import torch
import yaml
from torchmetrics.classification import Accuracy, AUROC, F1Score

from src.data.datamodule import CrackDataModule
from src.evaluation.metrics import (
    faithfulness_drop_from_csv,
    localization_accuracy_from_dirs,
)
from src.models.resnet50 import ResNet50Module
from src.models.vit import VisionTransformerModule


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def build_model(cfg):
    if 'vit' in cfg.get('model', '').lower():
        return VisionTransformerModule(cfg)
    return ResNet50Module(cfg)


def get_dataloader(datamodule, split: str):
    split = split.lower()
    if split == 'train':
        return datamodule.train_dataloader()
    if split == 'val':
        return datamodule.val_dataloader()
    if split == 'test':
        return datamodule.test_dataloader()
    if split == 'robust':
        return datamodule.robustness_dataloader()
    raise ValueError(f'Unsupported split "{split}". Choose from train|val|test|robust.')


def compute_classification_metrics(model, dataloader, device):
    acc = Accuracy(task='binary').to(device)
    f1 = F1Score(task='binary').to(device)
    auc = AUROC(task='binary').to(device)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            acc(preds, y)
            f1(preds, y)
            auc(probs, y)

    return {
        'accuracy': acc.compute().item(),
        'f1': f1.compute().item(),
        'auc': auc.compute().item(),
    }


def main(args):
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = CrackDataModule(
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        robustness_split=config.get('robustness_split', 0.1)
    )
    datamodule.setup()

    dataloader = get_dataloader(datamodule, args.split)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    classification_metrics = compute_classification_metrics(model, dataloader, device)
    localization_score = localization_accuracy_from_dirs(args.cam_dir, args.mask_dir, threshold=args.localization_threshold)
    faithfulness_score = faithfulness_drop_from_csv(args.faithfulness_csv)

    results = {
        'split': args.split,
        'accuracy': classification_metrics['accuracy'],
        'f1': classification_metrics['f1'],
        'auc': classification_metrics['auc'],
        'localization_accuracy': localization_score,
        'faithfulness': faithfulness_score,
    }

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'Metrics saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained crack detection models.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config used for training')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.ckpt)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split: train|val|test|robust')
    parser.add_argument('--cam-dir', type=str, default=None, help='Directory with saved Grad-CAM/attention maps')
    parser.add_argument('--mask-dir', type=str, default=None, help='Directory with ground-truth localization masks')
    parser.add_argument('--localization-threshold', type=float, default=0.5, help='Threshold for localization IoU')
    parser.add_argument('--faithfulness-csv', type=str, default=None, help='CSV with original_prob and perturbed_prob columns')
    parser.add_argument('--output', type=str, default=None, help='Optional path to save metrics JSON')
    args = parser.parse_args()
    main(args)