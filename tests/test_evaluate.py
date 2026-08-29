"""Tests for evaluate.evaluate_model against synthetic data with known predictions."""

import torch
import torch.nn as nn
from image_classification_cnn.evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset


class _PerfectClassifier(nn.Module):
    """Ignores the input and returns one-hot logits matching `fixed_labels`."""

    def __init__(self, fixed_labels: torch.Tensor, n_classes: int) -> None:
        super().__init__()
        self.fixed_labels = fixed_labels
        self.n_classes = n_classes
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = self.fixed_labels[self._call_count : self._call_count + x.size(0)]
        self._call_count += x.size(0)
        logits = torch.full((x.size(0), self.n_classes), -10.0)
        logits[torch.arange(x.size(0)), batch] = 10.0
        return logits


def test_evaluate_model_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    x = torch.zeros(8, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, labels), batch_size=4)
    model = _PerfectClassifier(labels, n_classes=4)

    metrics, y_true, y_pred = evaluate_model(model, loader, torch.device("cpu"))

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert y_true == labels.tolist()
    assert y_pred == labels.tolist()


def test_evaluate_model_known_error_rate():
    labels = torch.tensor([0, 0, 1, 1])
    wrong_preds = torch.tensor([0, 1, 1, 1])  # 1 mistake out of 4
    x = torch.zeros(4, 1, 28, 28)
    loader = DataLoader(TensorDataset(x, labels), batch_size=4)
    model = _PerfectClassifier(wrong_preds, n_classes=2)

    metrics, _, y_pred = evaluate_model(model, loader, torch.device("cpu"))

    assert metrics["accuracy"] == 0.75
    assert y_pred == wrong_preds.tolist()
