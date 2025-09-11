import torch, os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def device_select():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_curves(history, outpath):
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.plot(history['train_loss'], label='train_loss')
    ax.plot(history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training & Validation Loss')
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def plot_confusion(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def save_sample_predictions(images, labels, preds, classes, outpath, max_n=16):
    import torchvision
    n = min(len(images), max_n)
    imgs = images[:n].cpu()
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.9)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(grid.permute(1,2,0).numpy())
    ax.axis('off')
    title = " | ".join([f"T:{classes[int(labels[i])]} P:{classes[int(preds[i])]}" for i in range(n)])
    ax.set_title(title[:120])
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)
