import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small CNN: 2x (Conv -> ReLU -> MaxPool) + FC head.

    Input is assumed to be square, ``img_size`` x ``img_size``. The flattened
    feature size after the two 2x2 max-pools is computed from a dummy forward
    pass rather than assumed from ``in_ch``, so any (in_ch, img_size)
    combination works correctly -- not just the MNIST (1, 28) and CIFAR10
    (3, 32) cases this model ships with.
    """

    def __init__(self, in_ch: int = 1, n_classes: int = 10, img_size: int = 28) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_size, img_size)
            flat_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
