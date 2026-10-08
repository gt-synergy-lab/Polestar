# Evaluation

## Text benchmarks

```bash
python -m pip install -e '.[eval]'
python evaluation/run.py --config configs/llada8b.json --task gsm8k --output-path results/llada8b-gsm8k-256
```

| Task argument | Benchmark | Few-shot |
|---|---|---|
| `gsm8k` | GSM8K | 5 |
| `minerva_math` | MATH | 4 |
| `humaneval` | HumanEval | 0 |
| `mbpp` | MBPP | 3 |

Replace the model preset with `configs/llada15.json` or `configs/dream7b.json` to evaluate the other text models. Use `--max-new-tokens 512` for the longer generation setting. Presets and the task protocol are in `configs/evaluation/text.json`.

For a smoke run, add `--limit 8` and use a separate output path. A fraction such as `--limit 0.1` evaluates 10% of each selected task. Omitting `--limit` evaluates the complete task.

Add `--predict-only` to save generated responses without computing task scores. For example:

```bash
python evaluation/run.py --config configs/llada8b.json --task gsm8k \
  --limit 8 --predict-only --output-path results/llada8b-gsm8k-smoke
```

## Code generation and scoring

Code tasks execute generated programs. Pass `--allow-code-execution` when intentionally running them in your evaluation environment:

```bash
python evaluation/run.py --config configs/dream7b.json --task humaneval \
  --allow-code-execution --output-path results/dream-humaneval-256
python evaluation/postprocess_humaneval.py results/dream-humaneval-256/samples_humaneval.jsonl
```

The HumanEval postprocessor extracts/sanitizes the function completion before scoring. For MBPP:

```bash
python evaluation/run.py --config configs/llada8b.json --task mbpp \
  --allow-code-execution --output-path results/llada8b-mbpp-256
python evaluation/postprocess_mbpp.py results/llada8b-mbpp-256/samples_mbpp.jsonl --run-tests
```

Keep raw and postprocessed scores together with their sample files. The runner writes `results.json` and `samples_<task>.jsonl` to a new output directory and will not overwrite an existing results file.

## Image benchmarks

Use a separate environment with the vision evaluation extra:

```bash
python -m pip install -e '.[vision-eval]'
python evaluation/run.py --config configs/llada-v.json --task mathvista_testmini \
  --output-path results/llada-v-mathvista
python evaluation/run.py --config configs/llada-v.json --task mathverse_testmini_vision \
  --output-path results/llada-v-mathverse
```

The official task scorers use an OpenAI model to extract/judge answers. Set `OPENAI_API_KEY` in your environment for these evaluation commands. Scoring model settings are in `configs/evaluation/vision.json`.

To check image-benchmark generation and sample export without calling the answer scorer, use prediction mode:

```bash
python evaluation/run.py --config configs/llada-v.json --task mathvista_testmini \
  --limit 8 --predict-only --output-path results/llada-v-mathvista-smoke
```

Prediction mode and the standalone generation examples do not use the scoring API.

## Results and measurements

The [paper](https://arxiv.org/pdf/2607.14107) contains accuracy, tokens-per-forward (TPF), and tokens-per-second (TPS) across model/task settings. Record model/checkpoint, preset, seed, task/harness version, few-shot count, generation length, batch size, hardware, and scoring method with each run. Generation accounting excludes prompt tokens; code-task scores use their task's postprocessing and tests. Keep generation timing distinct from model download, task loading, and external answer scoring.
