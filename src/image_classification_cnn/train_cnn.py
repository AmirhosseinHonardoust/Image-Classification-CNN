import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, load_config_file
from .datasets import build_loaders
from .model import SimpleCNN
from .utils import data_root, device_select, plot_curves, save_sample_predictions, setup_logging

logger = setup_logging()


def get_data(
    dataset: str, batch_size: int, num_workers: int = 2
) -> tuple[DataLoader, DataLoader, int, int, list[str], int]:
    """Build train/val DataLoaders for the given dataset ('mnist' or 'cifar10').

    Returns (train_loader, val_loader, in_channels, n_classes, class_names, img_size).
    """
    root = data_root()
    train_loader, spec = build_loaders(dataset, root, batch_size, num_workers, train=True)
    val_loader, _ = build_loaders(dataset, root, batch_size, num_workers, train=False)
    return (
        train_loader,
        val_loader,
        spec.in_ch,
        len(spec.class_names),
        spec.class_names,
        spec.img_size,
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of predictions in `logits` (argmax over classes) matching `y`."""
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    outdir: str,
    lr: float = 1e-3,
    patience: int = 3,
    classes: list[str] | None = None,
) -> tuple[str, float, dict[str, list[float]]]:
    """Train `model` with early stopping on validation accuracy.

    Saves the best checkpoint (by val_acc) to `outdir/best_cnn.pth`, a
    sample-predictions image whenever the checkpoint improves, and a
    training-curves plot at the end.

    `classes` labels the sample-predictions image. If omitted, digit
    strings ("0", "1", ...) sized to the model's actual output classes
    are used instead.

    Returns (best_checkpoint_path, best_val_acc, history).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_path = os.path.join(outdir, "best_cnn.pth")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_acc += accuracy(logits.detach(), y) * x.size(0)
            n += x.size(0)
        train_loss = total_loss / n
        train_acc = total_acc / n
        model.eval()
        vloss, vacc, vn = 0.0, 0.0, 0
        first_batch_imgs: torch.Tensor | None = None
        first_batch_lbls: torch.Tensor | None = None
        first_batch_preds: torch.Tensor | None = None
        first_batch_n_classes = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]")):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                vloss += loss.item() * x.size(0)
                vacc += accuracy(logits, y) * x.size(0)
                vn += x.size(0)
                if i == 0:
                    first_batch_imgs = x.clone()
                    first_batch_lbls = y.clone()
                    first_batch_preds = logits.argmax(1).clone()
                    first_batch_n_classes = logits.size(1)
        val_loss = vloss / vn
        val_acc = vacc / vn
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, best_path)
            stale = 0
            assert first_batch_imgs is not None
            assert first_batch_lbls is not None
            assert first_batch_preds is not None
            save_sample_predictions(
                first_batch_imgs.cpu(),
                first_batch_lbls.cpu(),
                first_batch_preds.cpu(),
                classes=(
                    classes
                    if classes is not None
                    else [str(i) for i in range(first_batch_n_classes)]
                ),
                outpath=os.path.join(outdir, "sample_predictions.png"),
            )
        else:
            stale += 1
        logger.info(
            "[epoch %d] train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            ep,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )
        if stale >= patience:
            logger.info("Early stopping at epoch %d. Best val_acc=%.4f", ep, best_val_acc)
            break
    plot_curves(history, os.path.join(outdir, "training_curves.png"))
    return best_path, best_val_acc, history


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Merge CLI args over an optional --config JSON file over TrainConfig defaults.

    Precedence: explicit CLI flag > --config file value > TrainConfig default.
    """
    file_values = load_config_file(args.config)
    defaults = TrainConfig()
    return TrainConfig(
        dataset=args.dataset or file_values.get("dataset", defaults.dataset),
        epochs=(
            args.epochs if args.epochs is not None else file_values.get("epochs", defaults.epochs)
        ),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else file_values.get("batch_size", defaults.batch_size)
        ),
        lr=args.lr if args.lr is not None else file_values.get("lr", defaults.lr),
        outdir=args.outdir or file_values.get("outdir", defaults.outdir),
        patience=(
            args.patience
            if args.patience is not None
            else file_values.get("patience", defaults.patience)
        ),
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
    ap.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--patience", type=int, default=None, help="Early-stopping patience (epochs)")
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
    train_loader, val_loader, in_ch, n_classes, classes, img_size = get_data(
        cfg.dataset, cfg.batch_size, cfg.num_workers
    )
    model = SimpleCNN(in_ch=in_ch, n_classes=n_classes, img_size=img_size).to(device)

    best_path, best_val_acc, history = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        cfg.epochs,
        cfg.outdir,
        lr=cfg.lr,
        patience=cfg.patience,
        classes=classes,
    )
    with open(os.path.join(cfg.outdir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_val_acc, "epochs": len(history["train_loss"])}, f, indent=2)
    logger.info("[OK] Training complete. Best model @ %s (val_acc=%.4f)", best_path, best_val_acc)


if __name__ == "__main__":
    main()
