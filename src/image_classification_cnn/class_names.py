"""Shared, dataset-related constants used by ``datasets.py``.

Kept separate so nothing needs to import train_cnn.py (and its
argparse/tqdm/optimizer imports and module-level logger setup) just to get
the CIFAR10 class names.
"""

from __future__ import annotations

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
