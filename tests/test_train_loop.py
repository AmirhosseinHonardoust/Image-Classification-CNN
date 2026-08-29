"""End-to-end test of train_cnn.train_loop against a tiny synthetic dataset.

Uses random tensors instead of real MNIST/CIFAR10 so this runs fast and
offline, but exercises the exact training/validation/checkpoint/early-stop
code path used in production.
"""

from pathlib import Path

import torch
from image_classification_cnn import train_cnn
from image_classification_cnn.model import SimpleCNN
from image_classification_cnn.train_cnn import train_loop
from torch.utils.data import DataLoader, TensorDataset


def _make_loader(n: int, in_ch: int, size: int, n_classes: int, batch_size: int = 8) -> DataLoader:
    x = torch.randn(n, in_ch, size, size)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_train_loop_runs_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    train_loader = _make_loader(n=32, in_ch=1, size=28, n_classes=10)
    val_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=10)
    model = SimpleCNN(in_ch=1, n_classes=10)

    best_path, best_val_acc, history = train_loop(
        model, train_loader, val_loader, torch.device("cpu"), epochs=2, outdir=str(tmp_path)
    )

    assert Path(best_path).exists()
    assert (tmp_path / "training_curves.png").exists()
    assert 0.0 <= best_val_acc <= 1.0
    assert len(history["train_loss"]) == 2
    assert len(history["val_acc"]) == 2


def test_train_loop_early_stopping(tmp_path):
    """With patience=1 and a model that can't improve, training stops early."""
    torch.manual_seed(0)
    train_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=10)
    val_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=10)
    model = SimpleCNN(in_ch=1, n_classes=10)

    _, _, history = train_loop(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        epochs=20,
        outdir=str(tmp_path),
        patience=1,
    )

    # Early stopping (patience=1) should trigger well before 20 epochs on
    # this tiny/noisy synthetic dataset.
    assert len(history["train_loss"]) < 20


def test_train_loop_deterministic_with_fixed_seed(tmp_path):
    """Same seed + same data => identical training history (regression guard)."""

    def run(outdir):
        torch.manual_seed(42)
        train_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=10, batch_size=4)
        val_loader = _make_loader(n=8, in_ch=1, size=28, n_classes=10, batch_size=4)
        torch.manual_seed(42)
        model = SimpleCNN(in_ch=1, n_classes=10)
        _, _, history = train_loop(
            model, train_loader, val_loader, torch.device("cpu"), epochs=1, outdir=outdir
        )
        return history

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    history_a = run(str(out_a))
    history_b = run(str(out_b))
    assert history_a == history_b


def test_train_loop_passes_provided_classes_to_sample_predictions(tmp_path, monkeypatch):
    """An explicit `classes` list reaches save_sample_predictions unchanged."""
    captured = {}

    def fake_save_sample_predictions(images, labels, preds, classes, outpath, max_n=16):
        captured["classes"] = classes

    monkeypatch.setattr(train_cnn, "save_sample_predictions", fake_save_sample_predictions)
    torch.manual_seed(0)
    train_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=4)
    # A large-enough val set makes "0 correct out of N" astronomically unlikely,
    # so the checkpoint (and sample-predictions save) reliably fires this epoch.
    val_loader = _make_loader(n=64, in_ch=1, size=28, n_classes=4)
    model = SimpleCNN(in_ch=1, n_classes=4)

    train_loop(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        epochs=1,
        outdir=str(tmp_path),
        classes=["a", "b", "c", "d"],
    )

    assert captured["classes"] == ["a", "b", "c", "d"]


def test_train_loop_default_classes_match_model_n_classes(tmp_path, monkeypatch):
    """Without an explicit `classes` list, sample-prediction labels should match
    the model's actual class count (regression guard: this used to be hardcoded
    to range(10), which silently mislabeled any model with n_classes != 10)."""
    captured = {}

    def fake_save_sample_predictions(images, labels, preds, classes, outpath, max_n=16):
        captured["classes"] = classes

    monkeypatch.setattr(train_cnn, "save_sample_predictions", fake_save_sample_predictions)
    torch.manual_seed(0)
    train_loader = _make_loader(n=16, in_ch=1, size=28, n_classes=4)
    val_loader = _make_loader(n=64, in_ch=1, size=28, n_classes=4)
    model = SimpleCNN(in_ch=1, n_classes=4)

    train_loop(model, train_loader, val_loader, torch.device("cpu"), epochs=1, outdir=str(tmp_path))

    assert captured["classes"] == ["0", "1", "2", "3"]
