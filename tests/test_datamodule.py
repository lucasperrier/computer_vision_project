import torch

from src.data.datamodule import CrackDataModule


def test_datamodule_outputs_shapes_and_labels():
    dm = CrackDataModule(batch_size=8, num_workers=0, verbose=False)
    dm.setup()

    x, y = next(iter(dm.test_dataloader()))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.ndim == 4
    assert x.shape[1:] == (3, 224, 224)
    assert y.dtype == torch.long

    # Ensure at least one class is present in the test set overall
    ys = []
    for _, yy in dm.test_dataloader():
        ys.append(yy)
    ys = torch.cat(ys)

    uniq = ys.unique(sorted=True).tolist()
    assert set(uniq).issubset({0, 1})
    assert len(set(uniq)) == 2, f"Expected both classes in test set, got unique={uniq}"