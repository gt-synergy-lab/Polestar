"""Selected scoring models must reach the actual per-task result functions."""

from types import FunctionType, SimpleNamespace

import pytest

from polestar.evaluation.vision import _apply_scoring_models


def function_body():
    return None


class Task:
    def __init__(self, family):
        self.config = {"task": f"{family}_testmini", "metadata": {"version": 1, "gpt_eval_model_name": "gpt-3.5-turbo"}}
        self.scorers = []
        self.namespaces = []
        functions = []
        for _ in range(4):
            scorer = SimpleNamespace(gpt_model="gpt-3.5-turbo")
            namespace = {f"{family}_evaluator": scorer, "config": {"metadata": {"gpt_eval_model_name": "gpt-3.5-turbo", "quick_extract": False}}}
            functions.append(FunctionType(function_body.__code__, namespace))
            self.scorers.append(scorer)
            self.namespaces.append(namespace)
        self.config["process_results"] = functions[0]
        self.aggregate = functions[1]
        self.config["doc_to_text"] = functions[2]
        self.config["doc_to_visual"] = functions[3]

    def get_config(self, key):
        return self.config.get(key)

    def set_config(self, key, value):
        self.config[key] = value

    def aggregation(self):
        return {"gpt_eval_score": self.aggregate}


@pytest.mark.parametrize("family", ["mathvista", "mathverse"])
def test_resolved_process_and_aggregation_use_selected_scorer(family):
    selected = Task(family)
    other = Task("unrelated")
    process = selected.config["process_results"]
    aggregate = selected.aggregate
    _apply_scoring_models({"group": {"selected": selected, "other": other}}, {family: "gpt-4o"})
    assert [scorer.gpt_model for scorer in selected.scorers] == ["gpt-4o"] * 4
    assert selected.config["metadata"] == {"version": 1, "gpt_eval_model_name": "gpt-4o"}
    assert selected.config["process_results"] is process
    assert selected.aggregate is aggregate
    assert all(namespace["config"]["metadata"] == {"gpt_eval_model_name": "gpt-4o", "quick_extract": False} for namespace in selected.namespaces)
    assert all(scorer.gpt_model == "gpt-3.5-turbo" for scorer in other.scorers)
