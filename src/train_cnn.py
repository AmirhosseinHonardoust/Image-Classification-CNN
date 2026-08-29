import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import data_root, device_select, plot_curves, save_sample_predictions


def get_data(dataset, batch_size):
    root = data_root()
    if dataset.lower() == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
        in_ch, n_classes = 1, 10
    elif dataset.lower() == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        in_ch, n_classes = 3, 10
    else:
        raise ValueError("dataset must be 'mnist' or 'cifar10'")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    classes = [str(i) for i in range(n_classes)]
    return train_loader, val_loader, in_ch, n_classes, classes


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_loop(model, train_loader, val_loader, device, epochs, outdir, lr=1e-3, patience=3):
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
                classes=[str(i) for i in range(10)],
                outpath=os.path.join(outdir, "sample_predictions.png"),
            )
        else:
            stale += 1
        print(
            f"[epoch {ep}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if stale >= patience:
            print(f"Early stopping at epoch {ep}. Best val_acc={best_val_acc:.4f}")
            break
    plot_curves(history, os.path.join(outdir, "training_curves.png"))
    return best_path, best_val_acc, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = device_select()
    train_loader, val_loader, in_ch, n_classes, classes = get_data(args.dataset, args.batch_size)
    model = SimpleCNN(in_ch=in_ch, n_classes=n_classes).to(device)

    best_path, best_val_acc, history = train_loop(
        model, train_loader, val_loader, device, args.epochs, args.outdir, lr=args.lr
    )
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_val_acc, "epochs": len(history["train_loss"])}, f, indent=2)
    print(f"[OK] Training complete. Best model @ {best_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
