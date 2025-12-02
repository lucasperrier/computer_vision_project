import pytorch_lightning as pl
import torch
import torch.nn as nn
import timm
from torchmetrics import Accuracy, F1Score, AUROC
from typing import Any, Dict


class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)

        self.learning_rate = self.hparams.get('learning_rate', 1e-4)
        self.weight_decay = self.hparams.get('weight_decay', 1e-5)
        self.scheduler_name = self.hparams.get('scheduler')
        self.optimizer_name = self.hparams.get('optimizer', 'adamw').lower()

        num_classes = self.hparams.get('num_classes', 2)
        model_name = self.hparams.get('model', 'vit_base_patch16_224')
        pretrained = self.hparams.get('pretrained', True)

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        freeze_layers = self.hparams.get('freeze_layers', 0)
        if freeze_layers:
            self._freeze_layers(freeze_layers)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.train_auc = AUROC(task='binary')

        self.val_acc = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_auc = AUROC(task='binary')

        self.test_acc = Accuracy(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_auc = AUROC(task='binary')

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        if stage == 'train':
            acc_metric = self.train_acc
            f1_metric = self.train_f1
            auc_metric = self.train_auc
        elif stage == 'val':
            acc_metric = self.val_acc
            f1_metric = self.val_f1
            auc_metric = self.val_auc
        elif stage == 'test':
            acc_metric = self.test_acc
            f1_metric = self.test_f1
            auc_metric = self.test_auc

        acc_metric(preds, y)
        f1_metric(preds, y)
        auc_metric(probs, y)

        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=(stage != 'train'))
        self.log(f'{stage}_acc', acc_metric, prog_bar=True)
        self.log(f'{stage}_f1', f1_metric)
        self.log(f'{stage}_auc', auc_metric)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, 'test')

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.get('epochs', 10)
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer

    def _freeze_layers(self, num_layers: int):
        if not hasattr(self.model, 'blocks'):
            return
        for idx, block in enumerate(self.model.blocks):
            if idx < num_layers:
                for param in block.parameters():
                    param.requires_grad = False