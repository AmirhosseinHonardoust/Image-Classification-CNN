"""Single source of truth for the datasets this project supports.

Both ``train_cnn.py`` and ``evaluate.py`` need to build MNIST/CIFAR10
DataLoaders; before this module they each had their own copy of that
branching logic. Centralizing it here means there's exactly one place
that knows a dataset's torchvision class, input channels, image size,
class names, and transforms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose

from .class_names import CIFAR10_CLASSES

VALID_DATASETS = ("mnist", "cifar10")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to build DataLoaders for one supported dataset."""

    dataset_cls: Callable[..., Dataset]
    in_ch: int
    img_size: int
    class_names: list[str]
    train_transform: Compose
    test_transform: Compose = field(
        default_factory=lambda: transforms.Compose([transforms.ToTensor()])
    )


_REGISTRY: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        dataset_cls=datasets.MNIST,
        in_ch=1,
        img_size=28,
        class_names=[str(i) for i in range(10)],
        train_transform=transforms.Compose([transforms.ToTensor()]),
    ),
    "cifar10": DatasetSpec(
        dataset_cls=datasets.CIFAR10,
        in_ch=3,
        img_size=32,
        class_names=CIFAR10_CLASSES,
        train_transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    ),
}


def get_spec(dataset: str) -> DatasetSpec:
    """Look up the :class:`DatasetSpec` for ``dataset`` (case-insensitive)."""
    key = dataset.lower()
    if key not in _REGISTRY:
        raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {dataset!r}")
    return _REGISTRY[key]


def build_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    num_workers: int = 2,
    train: bool = True,
) -> tuple[DataLoader, DatasetSpec]:
    """Build a single DataLoader (train or test split) plus its DatasetSpec.

    ``train=True`` uses the dataset's augmented train transform and shuffles;
    ``train=False`` uses the plain test transform and does not shuffle.
    """
    spec = get_spec(dataset)
    tfm = spec.train_transform if train else spec.test_transform
    ds = spec.dataset_cls(root=root, train=train, download=True, transform=tfm)
    loader: DataLoader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, spec
