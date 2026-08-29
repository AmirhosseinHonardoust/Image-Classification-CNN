import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .config import EvalConfig, load_config_file
from .datasets import build_loaders
from .model import SimpleCNN
from .utils import data_root, device_select, plot_confusion, setup_logging

logger = setup_logging()


def get_test_loader(
    dataset: str, batch_size: int, num_workers: int = 2
) -> tuple[DataLoader, int, list[str], int]:
    """Build the test DataLoader for the given dataset ('mnist' or 'cifar10').

    Returns (loader, in_channels, class_names, img_size).
    """
    root = data_root()
    loader, spec = build_loaders(dataset, root, batch_size, num_workers, train=False)
    return loader, spec.in_ch, spec.class_names, spec.img_size


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], list[int], list[int]]:
    """Run `model` over `loader` and compute accuracy/precision/recall/F1 (macro).

    Returns (metrics, y_true, y_pred) so callers can also plot a confusion
    matrix from the same predictions without re-running inference.
    """
    model.eval()
    all_y: list[int] = []
    all_p: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(preds.cpu().tolist())

    acc = accuracy_score(all_y, all_p)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y, all_p, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }
    return metrics, all_y, all_p


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Merge CLI args over an optional --config JSON file over EvalConfig defaults.

    Precedence: explicit CLI flag > --config file value > EvalConfig default.
    """
    file_values = load_config_file(args.config)
    defaults = EvalConfig(model=args.model or file_values.get("model", ""))
    return EvalConfig(
        model=args.model or file_values.get("model", defaults.model),
        dataset=args.dataset or file_values.get("dataset", defaults.dataset),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else file_values.get("batch_size", defaults.batch_size)
        ),
        outdir=args.outdir or file_values.get("outdir", defaults.outdir),
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else file_values.get("num_workers", defaults.num_workers)
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default=None, help="Optional JSON file overriding defaults below"
    )
    ap.add_argument("--model", default=None, help="Path to a checkpoint saved by train_cnn.py")
    ap.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader worker processes (default: 2)"
    )
    args = ap.parse_args()

    try:
        cfg = build_config(args)
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(1)

    os.makedirs(cfg.outdir, exist_ok=True)
    device = device_select()

    loader, in_ch, classes, img_size = get_test_loader(cfg.dataset, cfg.batch_size, cfg.num_workers)
    model = SimpleCNN(in_ch=in_ch, n_classes=len(classes), img_size=img_size).to(device)
    state = torch.load(cfg.model, map_location=device, weights_only=True)
    model.load_state_dict(
        state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    )

    metrics, all_y, all_p = evaluate_model(model, loader, device)

    with open(os.path.join(cfg.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion(all_y, all_p, classes, os.path.join(cfg.outdir, "confusion_matrix.png"))
    logger.info(
        "[OK] Evaluation complete. Acc=%.4f  F1(macro)=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
    )


if __name__ == "__main__":
    main()
