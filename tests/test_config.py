import json

import pytest

from polestar import PolestarConfig


@pytest.mark.parametrize("filename", ["llada8b.json", "llada15.json", "dream7b.json", "llada-v.json"])
def test_model_presets(filename):
    config = PolestarConfig.from_json("configs/" + filename)
    assert config.generation["block_length"] == 32
    assert config.generation["threshold"] == 0.9
    assert config.generation["threshold_early"] == 0.7


def test_invalid_block_schedule():
    with pytest.raises(ValueError, match="divisible"):
        PolestarConfig("llada", "checkpoint", {"max_new_tokens": 33, "block_length": 32})


def test_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        PolestarConfig("unknown", "checkpoint")


def test_configuration_roundtrip(tmp_path):
    data = {"model_type": "dream", "model_id": "checkpoint", "generation": {"max_new_tokens": 512, "steps": 512, "block_length": 32}}
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data))
    assert PolestarConfig.from_json(path).generation == data["generation"]
