"""Evaluation controls should reach each harness without changing model settings."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "polestar_evaluation_runner", Path(__file__).parents[1] / "evaluation" / "run.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("value, expected", [("1", 1), ("8", 8), ("1.0", 1), ("0.1", 0.1)])
def test_count_or_fraction_limit(value, expected):
    result = runner.parse_limit(value)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan", "inf", "all"])
def test_invalid_limit(value):
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_limit(value)


@pytest.mark.parametrize(
    "model_type, task, token_available",
    [
        ("llada", "gsm8k", False),
        ("dream", "gsm8k", False),
        ("llada_v", "mathvista_testmini", False),
        ("llada_v", "mathvista_testmini", True),
        ("llada_v", "mathverse_testmini_vision", False),
    ],
)
def test_prediction_mode_routes_to_harness_and_exports_samples(monkeypatch, tmp_path, model_type, task, token_available):
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {}, "samples": {task: [{"resps": [["generated answer"]]}]}}

    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)))
    monkeypatch.setitem(
        sys.modules, "torch",
        SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "lmms_eval", SimpleNamespace(evaluator=SimpleNamespace(simple_evaluate=evaluate)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: "configured" if token_available else None))
    monkeypatch.setitem(sys.modules, "polestar.evaluation.llada", SimpleNamespace(LLaDAEvalHarness=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "polestar.evaluation.dream", SimpleNamespace(Dream=lambda **kwargs: kwargs))
    monkeypatch.setitem(
        sys.modules, "polestar.evaluation.vision",
        SimpleNamespace(register_model=lambda settings: None, runtime_stats=lambda: {"total_nfe": 1}, create_task_manager=lambda settings: "configured task manager"),
    )
    preset = tmp_path / "model.json"
    preset.write_text(json.dumps({"model_type": model_type, "model_id": "checkpoint"}))
    output = tmp_path / "run"
    monkeypatch.chdir(Path(__file__).parents[1])
    monkeypatch.setattr(
        sys, "argv",
        ["run.py", "--config", str(preset), "--task", task, "--limit", "0.1", "--predict-only", "--seed", "17", "--output-path", str(output)],
    )
    runner.main()

    assert len(calls) == 1
    assert calls[0]["predict_only"] is True
    assert calls[0]["limit"] == 0.1
    assert calls[0]["log_samples"] is True
    assert all(calls[0][key] == 17 for key in ["random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"])
    if model_type == "llada_v":
        assert calls[0]["cli_args"].output_path == str(output)
        assert "batch_size=" not in calls[0]["model_args"]
        assert calls[0]["batch_size"] == 1
        assert calls[0]["task_manager"] == "configured task manager"
        expected_task = {"task": task, "dataset_kwargs": {"token": False}} if task == "mathvista_testmini" and not token_available else task
        assert calls[0]["tasks"] == [expected_task]
    metadata = json.loads((output / "results.json").read_text())["polestar"]
    assert metadata["predict_only"] is True
    assert metadata["limit"] == 0.1
    assert metadata["seed"] == 17
    samples = [json.loads(row) for row in (output / f"samples_{task}.jsonl").read_text().splitlines()]
    assert samples == [{"resps": [["generated answer"]]}]
