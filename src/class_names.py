"""Shared, dataset-related constants used by both train_cnn.py and evaluate.py.

Kept separate from train_cnn.py so evaluate.py doesn't need to import the
training script (and its argparse/tqdm/optimizer imports and module-level
logger setup) just to get the CIFAR10 class names.

Named ``class_names`` rather than ``datasets`` to avoid colliding with the
``from torchvision import datasets`` import used in train_cnn.py/evaluate.py.
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
