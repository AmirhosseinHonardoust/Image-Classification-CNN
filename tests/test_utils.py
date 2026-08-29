from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import torch  # noqa: E402
from image_classification_cnn.train_cnn import accuracy  # noqa: E402
from image_classification_cnn.utils import (  # noqa: E402
    data_root,
    device_select,
    plot_confusion,
    plot_curves,
)


def test_device_select_returns_torch_device():
    device = device_select()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_data_root_points_at_repo_data_dir():
    root = data_root()
    assert Path(root).name == "data"
    assert Path(root).parent.name != "src"


def test_accuracy_all_correct():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    y = torch.tensor([0, 1])
    assert accuracy(logits, y) == 1.0


def test_accuracy_all_wrong():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    y = torch.tensor([1, 0])
    assert accuracy(logits, y) == 0.0


def test_plot_curves_writes_file(tmp_path):
    history = {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}
    outpath = tmp_path / "curves.png"
    plot_curves(history, str(outpath))
    assert outpath.exists()


def test_plot_confusion_writes_file(tmp_path):
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    outpath = tmp_path / "confusion.png"
    plot_confusion(y_true, y_pred, classes=["0", "1"], outpath=str(outpath))
    assert outpath.exists()
