# test_resnet.py
import torch
import pytorch_lightning as pl
from src.models.resnet50 import ResNet50Module
from src.data.datamodule import CrackDataModule

# Instantiate
model = ResNet50Module()
dm = CrackDataModule(batch_size=16)
dm.setup()

# Get a batch
train_dl = dm.train_dataloader()
batch = next(iter(train_dl))
x, y = batch

# Forward pass
with torch.no_grad():
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}, Labels: {y}")

# Quick training step (optional, for integration check)
trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)
trainer.fit(model, dm)

# Run validation to obtain scores
trainer = pl.Trainer(logger=False, enable_checkpointing=False)  # Minimal trainer for validation
results = trainer.validate(model, dm.val_dataloader())
print("Validation Scores:", results)
print("Model and datamodule integration successful.")