"""Typed, validated configuration for the training and evaluation scripts.

Defaults here match the original argparse defaults exactly, so omitting
both CLI flags and --config reproduces prior behavior unchanged. When a
--config JSON file is given, its values act as new defaults; any CLI flag
that is explicitly passed still takes precedence over both the config
file and these dataclass defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_DATASETS = ("mnist", "cifar10")


@dataclass
class TrainConfig:
    dataset: str = "mnist"
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    outdir: str = "outputs"
    patience: int = 3

    def __post_init__(self) -> None:
        if self.dataset.lower() not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {self.dataset!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")


@dataclass
class EvalConfig:
    model: str
    dataset: str = "mnist"
    batch_size: int = 128
    outdir: str = "outputs"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model path must be provided")
        if self.dataset.lower() not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {VALID_DATASETS}, got {self.dataset!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


def load_config_file(path: str | None) -> dict[str, Any]:
    """Read a JSON config file into a dict. Returns {} if path is None."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config file {path!r} must contain a JSON object")
    return data
