import argparse
import json
import os

import torch
from model import SimpleCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import data_root, device_select, plot_confusion


def get_test_loader(dataset, batch_size):
    root = data_root()
    if dataset.lower() == "mnist":
        tfm = transforms.ToTensor()
        ds = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
        in_ch, classes = 1, [str(i) for i in range(10)]
    elif dataset.lower() == "cifar10":
        tfm = transforms.ToTensor()
        ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
        in_ch, classes = 3, [str(i) for i in range(10)]
    else:
        raise ValueError("dataset must be 'mnist' or 'cifar10'")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader, in_ch, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = device_select()

    loader, in_ch, classes = get_test_loader(args.dataset, args.batch_size)
    model = SimpleCNN(in_ch=in_ch, n_classes=len(classes)).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(
        state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    )
    model.eval()

    all_y, all_p = [], []
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

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(
            {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1},
            f,
            indent=2,
        )

    plot_confusion(all_y, all_p, classes, os.path.join(args.outdir, "confusion_matrix.png"))
    print(f"[OK] Evaluation complete. Acc={acc:.4f}  F1(macro)={f1:.4f}")


if __name__ == "__main__":
    main()
