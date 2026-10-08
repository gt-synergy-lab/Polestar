# Models

## LLaDA

LLaDA-8B-Instruct and LLaDA-1.5 share the implementation in `src/polestar/models/llada/`. Their presets select the checkpoint without duplicating model code.

```bash
python examples/generate.py --config configs/llada8b.json --prompt "What is representation drift?"
python examples/generate.py --config configs/llada15.json --prompt "What is representation drift?"
```

## Dream

Dream uses its own attention, token alignment, and diffusion generation implementation in `src/polestar/models/dream/`.

```bash
python examples/generate.py --config configs/dream7b.json --prompt "Explain how KV caching reduces computation."
```

## LLaDA-V

LLaDA-V combines an image encoder/projector with the language model. Polestar's integration keeps image-token preparation and the image-conditioned generation hook together.

```bash
python -m pip install -e '.[vision]'
python examples/image_generation.py --config configs/llada-v.json \
  --image path/to/image.jpg --prompt "Describe the objects and their relationships."
```

The image example accepts a local JPEG/PNG or other Pillow-supported image. It inserts the image token, processes the image with the checkpoint's processor, and generates a response. The checkpoint's vision tower is loaded alongside its language model.

## Presets

`model_type` selects `llada`, `dream`, or `llada_v`; `model_id` selects a Hugging Face checkpoint or local directory. The `generation` object sets response length, steps, block length, temperature, and confidence thresholds. Defaults use generation length 256 and blocks of 32.

For a length of 512, set both `max_new_tokens` and `steps` to 512. Length must be divisible by block length, and steps by the number of blocks. Public examples use one sequence per call. `--seed` controls the example's random initialization.
