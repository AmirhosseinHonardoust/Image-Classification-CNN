"""Tests for the CLI-flag / --config-file / default precedence in each script's build_config()."""

import argparse
import json

from image_classification_cnn import evaluate, train_cnn


def _train_args(**overrides):
    base = {
        "config": None,
        "dataset": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "outdir": None,
        "patience": None,
        "num_workers": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_train_build_config_uses_defaults_when_nothing_given():
    cfg = train_cnn.build_config(_train_args())
    assert cfg.epochs == 10
    assert cfg.batch_size == 64


def test_train_build_config_cli_overrides_default():
    cfg = train_cnn.build_config(_train_args(epochs=3))
    assert cfg.epochs == 3
    assert cfg.batch_size == 64  # untouched flags stay at default


def test_train_build_config_file_overrides_default(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"epochs": 7, "batch_size": 32}))
    cfg = train_cnn.build_config(_train_args(config=str(config_path)))
    assert cfg.epochs == 7
    assert cfg.batch_size == 32


def test_train_build_config_cli_beats_file(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"epochs": 7}))
    cfg = train_cnn.build_config(_train_args(config=str(config_path), epochs=99))
    assert cfg.epochs == 99  # explicit CLI flag wins over the config file


def test_train_build_config_num_workers_cli_beats_file(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"num_workers": 4}))
    cfg = train_cnn.build_config(_train_args(config=str(config_path), num_workers=0))
    assert cfg.num_workers == 0


def _eval_args(**overrides):
    base = {
        "config": None,
        "model": None,
        "dataset": None,
        "batch_size": None,
        "outdir": None,
        "num_workers": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_eval_build_config_cli_beats_file(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"model": "from_file.pth"}))
    cfg = evaluate.build_config(_eval_args(config=str(config_path), model="from_cli.pth"))
    assert cfg.model == "from_cli.pth"


def test_eval_build_config_file_overrides_default(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"model": "ckpt.pth", "batch_size": 256}))
    cfg = evaluate.build_config(_eval_args(config=str(config_path)))
    assert cfg.model == "ckpt.pth"
    assert cfg.batch_size == 256
