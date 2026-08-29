import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small CNN: 2x (Conv -> ReLU -> MaxPool) + FC head.

    Input is assumed to be square. ``in_ch=1`` (MNIST, 28x28) yields a 7x7
    feature map after two 2x2 max-pools; any other ``in_ch`` (e.g. CIFAR10,
    3x32x32) yields 8x8.
    """

    def __init__(self, in_ch: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat_dim = 64 * 7 * 7 if in_ch == 1 else 64 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
