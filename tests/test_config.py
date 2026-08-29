import json

import pytest
from image_classification_cnn.config import EvalConfig, TrainConfig, load_config_file


def test_train_config_defaults_match_original_argparse_defaults():
    cfg = TrainConfig()
    assert cfg.dataset == "mnist"
    assert cfg.epochs == 10
    assert cfg.batch_size == 64
    assert cfg.lr == 1e-3
    assert cfg.outdir == "outputs"
    assert cfg.patience == 3
    assert cfg.num_workers == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "not-a-real-dataset"},
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0.0},
        {"lr": -1.0},
        {"patience": 0},
        {"num_workers": -1},
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
