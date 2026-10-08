"""Run a Polestar benchmark with a checked-in model preset."""

import argparse
import json
import os
from pathlib import Path
import random

from polestar import PolestarConfig


def json_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Polestar")
    parser.add_argument("--config", default="configs/llada8b.json")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="Number of examples for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()

    config = PolestarConfig.from_json(args.config)
    if args.max_new_tokens is not None:
        config.generation.update(max_new_tokens=args.max_new_tokens, steps=args.max_new_tokens)
        config.__post_init__()
    output = Path(args.output_path)
    if (output / "results.json").exists():
        parser.error("results.json already exists; choose a new output directory")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive example count")

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if config.model_type == "llada_v":
        suite = json.loads(Path("configs/evaluation/vision.json").read_text())
        if args.task not in suite["tasks"]:
            parser.error("select a task from configs/evaluation/vision.json")
        from polestar.evaluation.vision import register_model, runtime_stats
        from lmms_eval import evaluator

        register_model(suite.get("scoring_models"))
        if args.max_new_tokens is not None:
            parser.error("set vision generation settings in its model preset")
        results = evaluator.simple_evaluate(
            model="polestar_llada_v",
            model_args=f"config_path={args.config},device={args.device},batch_size=1",
            tasks=[args.task],
            num_fewshot=suite.get("num_fewshot", 0),
            batch_size=1,
            limit=args.limit,
            log_samples=True,
        )
        results["polestar_runtime"] = runtime_stats()
    else:
        suite = json.loads(Path("configs/evaluation/text.json").read_text())
        if args.task not in suite["tasks"]:
            parser.error("select a task from configs/evaluation/text.json")
        task = suite["tasks"][args.task]
        if task.get("code_execution") and not args.allow_code_execution:
            parser.error("this task executes generated code; pass --allow-code-execution")
        if task.get("code_execution"):
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        settings = config.generation
        common = dict(
            batch_size=1,
            device=args.device,
            threshold=settings.get("threshold", 0.9),
            threshold_early=settings.get("threshold_early", 0.7),
            use_cache=True,
            dual_cache=True,
            use_cluster=True,
        )
        if config.model_type == "llada":
            from polestar.evaluation.llada import LLaDAEvalHarness
            model = LLaDAEvalHarness(
                model_path=config.model_id,
                gen_length=settings.get("max_new_tokens", 256),
                steps=settings.get("steps", 256),
                block_length=settings.get("block_length", 32),
                show_speed=True,
                **common,
            )
        else:
            from polestar.evaluation.dream import Dream
            model = Dream(
                pretrained=config.model_id,
                max_new_tokens=settings.get("max_new_tokens", 256),
                diffusion_steps=settings.get("steps", 256),
                block_length=settings.get("block_length", 32),
                temperature=settings.get("temperature", 0.0),
                add_bos_token=True,
                alg="confidence_threshold",
                **common,
            )
        import lm_eval
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=[args.task],
            num_fewshot=task["num_fewshot"],
            batch_size=1,
            limit=args.limit,
            log_samples=True,
            confirm_run_unsafe_code=args.allow_code_execution,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

    if results is None:
        raise RuntimeError("the evaluator returned no results")
    samples = results.pop("samples", {})
    results["polestar"] = {
        "model_preset": args.config,
        "model_id": config.model_id,
        "generation": config.generation,
        "seed": args.seed,
        "limit": args.limit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, default=json_value) + "\n")
    for task_name, rows in samples.items():
        with (output / f"samples_{task_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=json_value) + "\n")
    print(json.dumps(results.get("results", {}), indent=2, default=json_value))
    print(f"Saved results and samples to {output}")


if __name__ == "__main__":
    main()
