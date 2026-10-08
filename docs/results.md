# Paper results

Polestar uses representation drift to improve the accuracy–throughput trade-off across diffusion language and vision-language models. The results below are reported in [Polestar, arXiv:2607.14107](https://arxiv.org/pdf/2607.14107).

## Reading the metrics

- **Accuracy (%):** task accuracy for mathematics and multimodal benchmarks; pass@1 for HumanEval and MBPP.
- **TPF:** non-EOS generated response tokens divided by the number of model forward passes.
- **TPS:** the same generated-token count divided by end-to-end decoding time in seconds.

For MBPP, `±` denotes the accuracy error reported in Table 6 of the paper.

These tables use one NVIDIA A100 80GB and batch size 1. Throughput is measured across the complete fixed-length denoising trajectory, with prompt and EOS tokens excluded from the token count. Section D.1 of the paper describes the measurement protocol; Section D.6 reports additional GH200 results.

## LLaDA-8B-Instruct

Results from Table 1 (GSM8K, MATH, HumanEval) and Table 6 (MBPP).

| Benchmark | Few-shot | Generation length | Accuracy (%) ↑ | TPF ↑ | TPS ↑ |
|---|---:|---:|---:|---:|---:|
| GSM8K | 5 | 256 | 78.33 | 3.67 | 87.57 |
| GSM8K | 5 | 512 | 78.18 | 3.59 | 80.60 |
| MATH | 4 | 256 | 32.77 | 2.91 | 70.05 |
| MATH | 4 | 512 | 34.85 | 3.30 | 69.84 |
| HumanEval | 0 | 256 | 42.42 | 3.38 | 87.45 |
| HumanEval | 0 | 512 | 47.73 | 3.23 | 77.67 |
| MBPP | 3 | 256 | 31.63 ± 2.08 | 2.81 | 55.68 |
| MBPP | 3 | 512 | 14.12 ± 1.56 | 3.06 | 51.38 |

On GSM8K at generation length 256, Polestar reaches 87.57 TPS at 78.33% accuracy. The same table reports 47.88 TPS at 77.88% for Fast-dLLM, 49.15 TPS at 78.01% for Dynamic-dLLM, and 55.01 TPS at 77.58% for Elastic-Cache.

## Dream-7B-Instruct

Results from Table 1 (GSM8K, MATH, HumanEval) and Table 6 (MBPP).

| Benchmark | Few-shot | Generation length | Accuracy (%) ↑ | TPF ↑ | TPS ↑ |
|---|---:|---:|---:|---:|---:|
| GSM8K | 5 | 256 | 72.40 | 2.39 | 52.80 |
| GSM8K | 5 | 512 | 69.65 | 2.13 | 34.44 |
| MATH | 4 | 256 | 40.75 | 2.29 | 59.62 |
| MATH | 4 | 512 | 35.71 | 2.81 | 66.21 |
| HumanEval | 0 | 256 | 57.69 | 1.79 | 55.18 |
| HumanEval | 0 | 512 | 54.36 | 1.77 | 41.68 |
| MBPP | 3 | 256 | 54.91 ± 2.22 | 1.86 | 49.88 |
| MBPP | 3 | 512 | 51.36 ± 2.24 | 1.93 | 46.39 |

## LLaDA-1.5

Results from Table 2, which compares Polestar with parallel-decoding methods.

| Benchmark | Few-shot | Generation length | Accuracy (%) ↑ | TPF ↑ | TPS ↑ |
|---|---:|---:|---:|---:|---:|
| GSM8K | 5 | 256 | 81.06 | 3.40 | 79.65 |
| MBPP | 3 | 256 | 39.34 | 2.10 | 48.66 |

Table 7 provides additional LLaDA-1.5 results across the four text benchmarks and generation lengths 256 and 512.

## LLaDA-V

Results from Table 8 on MathVista and MathVerse.

| Benchmark | Accuracy (%) ↑ | TPF ↑ | TPS ↑ |
|---|---:|---:|---:|
| MathVista | 63.58 | 2.17 | 37.91 |
| MathVerse | 36.12 | 2.85 | 42.95 |

Polestar has the highest accuracy and TPS among the methods compared in Table 8 on both benchmarks.

## Run the benchmarks

The [evaluation guide](evaluation.md) provides model presets, few-shot settings, code-task postprocessing, multimodal scoring, and commands for both short checks and complete benchmark runs. Use the same model, task protocol, generation length, and hardware when comparing results.
