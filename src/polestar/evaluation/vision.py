"""Polestar LLaDA-V adapter for the MathVista and MathVerse evaluation tasks."""

from dataclasses import replace
import importlib
import time

_RUNTIME_STATS = {}


def runtime_stats():
    """Return the token, model-evaluation, and timing totals from the latest run."""
    return dict(_RUNTIME_STATS)


def register_model(scoring_models=None):
    """Register Polestar with lmms-eval when the vision evaluation is requested."""
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import MODEL_REGISTRY, register_model as register
    from lmms_eval.models import AVAILABLE_MODELS

    if "polestar_llada_v" in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY["polestar_llada_v"]
    else:
        model_class = _build_model_class(lmms)
        model_class = register("polestar_llada_v")(model_class)
    globals()["PolestarLLaDAV"] = model_class
    AVAILABLE_MODELS["polestar_llada_v"] = f"{__name__}.PolestarLLaDAV"

    if scoring_models:
        for task, scorer in scoring_models.items():
            if task not in {"mathvista", "mathverse"}:
                raise ValueError(f"Unsupported vision scoring task: {task}")
            task_utils = importlib.import_module(f"lmms_eval.tasks.{task}.utils")
            task_utils.config["metadata"]["gpt_eval_model_name"] = scorer
            getattr(task_utils, f"{task}_evaluator").gpt_model = scorer
    return model_class


def _build_model_class(base_class):
    class PolestarLLaDAV(base_class):
        def __init__(
            self,
            config_path="configs/llada-v.json",
            device="cuda",
            batch_size=1,
            think_mode="think",
            **kwargs,
        ):
            super().__init__()
            if kwargs:
                raise ValueError(f"Unexpected model arguments: {', '.join(sorted(kwargs))}")
            if int(batch_size) != 1:
                raise ValueError("LLaDA-V evaluation requires batch_size=1")
            if think_mode not in {"think", "no_think", ""}:
                raise ValueError("think_mode must be think, no_think, or an empty string")

            import torch
            from accelerate import Accelerator
            from polestar import PolestarConfig, load_model

            self.accelerator = Accelerator()
            if self.accelerator.num_processes > 1:
                device = f"cuda:{self.accelerator.local_process_index}"
            self._device = torch.device(device)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            self.polestar_config = PolestarConfig.from_json(config_path)
            if self.polestar_config.model_type != "llada_v":
                raise ValueError("Vision evaluation requires a LLaDA-V config")
            self._model, self._tokenizer = load_model(self.polestar_config, device=str(self._device))
            self._think_mode = think_mode
            torch.backends.cuda.matmul.allow_tf32 = True
            _RUNTIME_STATS.clear()
            _RUNTIME_STATS.update(total_time=0.0, total_tokens=0, total_nfe=0)
            self._eval_runtime_stats = _RUNTIME_STATS

        @property
        def model(self):
            return self._model

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def config(self):
            return self.model.config

        @property
        def device(self):
            return self._device

        @property
        def batch_size(self):
            return 1

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        @property
        def max_length(self):
            return getattr(self.config, "tokenizer_model_max_length", 16384)

        def tok_encode(self, text, left_truncate_len=None, add_special_tokens=False):
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return tokens[-left_truncate_len:] if left_truncate_len else tokens

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def loglikelihood(self, requests):
            raise NotImplementedError("Use the MathVista and MathVerse generation tasks")

        def generate_until_multi_round(self, requests):
            raise NotImplementedError("Use the MathVista and MathVerse single-round tasks")

        def generate_until(self, requests):
            from lmms_eval import utils
            from polestar.models.llada_v import generate
            from tqdm import tqdm

            def collate(request_args):
                return -len(self.tok_encode(request_args[0])), request_args[0]

            reordered = utils.Collator([request.args for request in requests], collate, grouping=True)
            outputs = []
            started = time.perf_counter()
            chunks = reordered.get_batched(n=1, batch_fn=None)
            for chunk in tqdm(chunks, total=len(requests), disable=self.rank != 0, desc="Polestar"):
                context, task_generation, doc_to_visual, doc_id, task, split = chunk[0]
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                if not isinstance(visuals, (list, tuple)) or len(visuals) != 1:
                    raise ValueError(f"{task} sample {doc_id} must provide exactly one image")

                settings = dict(self.polestar_config.generation)
                aliases = {"gen_length": "max_new_tokens", "gen_steps": "steps", "cfg": "cfg_scale"}
                accepted = {
                    "steps", "block_length", "temperature", "cfg_scale", "remasking",
                    "mask_id", "threshold", "threshold_early", "prefix_refresh_interval",
                    "cluster_num_centroids", "cluster_topk", "use_cluster", "early_stop",
                    "use_suffix_preunmask", "stopping_criteria", "generation_suffix",
                }
                for key, value in task_generation.items():
                    if key in aliases:
                        settings[aliases[key]] = value
                    elif key in accepted:
                        settings[key] = value
                settings.setdefault("stopping_criteria", ["<|eot_id|>"])
                generation_config = replace(self.polestar_config, generation=settings)

                think_mode = task_generation.get("think_mode", self._think_mode)
                question = context
                if "<image>" not in question and think_mode:
                    question += f" /{think_mode}"
                token_ids = generate(
                    self.model, self.tokenizer, question, generation_config, image=visuals[0]
                )
                response = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
                if response.endswith("."):
                    response = response[:-1]
                response = response.strip()
                outputs.append(response)
                self.cache_hook.add_partial("generate_until", (context, task_generation), response)
                _RUNTIME_STATS["total_tokens"] += int((token_ids != self.tokenizer.eos_token_id).sum().item())
                _RUNTIME_STATS["total_nfe"] += int(getattr(self.model, "_fast_dllm_last_stats", {}).get("nfe", 0))

            _RUNTIME_STATS["total_time"] += time.perf_counter() - started
            if _RUNTIME_STATS["total_time"] > 0:
                _RUNTIME_STATS["tokens_per_second"] = _RUNTIME_STATS["total_tokens"] / _RUNTIME_STATS["total_time"]
            if _RUNTIME_STATS["total_nfe"] > 0:
                _RUNTIME_STATS["tokens_per_forward_pass"] = _RUNTIME_STATS["total_tokens"] / _RUNTIME_STATS["total_nfe"]
            return reordered.get_original(outputs)

    return PolestarLLaDAV
