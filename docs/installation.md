# Installation

Polestar uses Python 3.10+, PyTorch 2.7.1, Transformers 4.49.0, and Accelerate 0.34.2. Generation uses a CUDA-capable NVIDIA GPU. The models contain 7–8B parameters; provision memory for the model weights, hidden-state cache, and sequence length.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Choose dependencies for your workload:

| Workload | Install |
|---|---|
| LLaDA / Dream generation | `python -m pip install -e .` |
| LLaDA-V image generation | `python -m pip install -e '.[vision]'` |
| Text benchmarks and code scoring | `python -m pip install -e '.[eval]'` |
| Multimodal benchmarks | `python -m pip install -e '.[vision-eval]'` |
| CPU checks | `python -m pip install -e '.[test]'` |

Text and multimodal evaluation environments can be installed separately: their harnesses pin different dependency versions. The model package is shared.

Set `HF_HOME` to a directory with space for checkpoint downloads. If a checkpoint requires authentication, use your Hugging Face environment or CLI login. Keep tokens in the environment rather than scripts or presets.

For a local checkpoint, replace `model_id` in a JSON preset with its directory. Run commands from the repository root so relative preset and output paths resolve consistently.

If PyTorch reports CUDA unavailable, check `nvidia-smi`, the installed PyTorch build, and the driver compatibility of your environment. On a cluster, run generation inside a compute-node allocation.
