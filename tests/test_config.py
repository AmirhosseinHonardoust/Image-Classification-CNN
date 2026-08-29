import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EvalConfig, TrainConfig, load_config_file  # noqa: E402


def test_train_config_defaults_match_original_argparse_defaults():
    cfg = TrainConfig()
    assert cfg.dataset == "mnist"
    assert cfg.epochs == 10
    assert cfg.batch_size == 64
    assert cfg.lr == 1e-3
    assert cfg.outdir == "outputs"
    assert cfg.patience == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "not-a-real-dataset"},
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": -1.0},
        {"patience": 0},
    ],
)
def test_train_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_eval_config_requires_model_path():
    with pytest.raises(ValueError):
        EvalConfig(model="")


def test_eval_config_rejects_invalid_dataset():
    with pytest.raises(ValueError):
        EvalConfig(model="ckpt.pth", dataset="imagenet")


def test_load_config_file_none_returns_empty_dict():
    assert load_config_file(None) == {}


def test_load_config_file_reads_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 5, "lr": 0.01}))
    assert load_config_file(str(path)) == {"epochs": 5, "lr": 0.01}


def test_load_config_file_rejects_non_object_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        load_config_file(str(path))
