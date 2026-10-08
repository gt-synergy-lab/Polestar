"""Load LLaDA-V and prepare image-conditioned Polestar generation."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from .constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .hooks.fast_dllm_hook import register_polestar_hook
from .mm_utils import process_images, tokenizer_image_token
from .model.language_model.llava_llada import LlavaLLaDAConfig, LlavaLLaDAModelLM

SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)


def load_model(config, device="cuda"):
    """Return the image-conditioned model and tokenizer for a Polestar config."""
    dtype_name = config.generation.get("dtype", "float16")
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = dtypes[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    model_config = LlavaLLaDAConfig.from_pretrained(config.model_id)
    model = LlavaLLaDAModelLM.from_pretrained(
        config.model_id,
        config=model_config,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if getattr(model.config, "mm_use_im_patch_token", True):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
    vision_tower.to(device=model.device, dtype=dtype)
    model._polestar_image_processor = vision_tower.image_processor
    model._polestar_generation_hook = register_polestar_hook(model)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, config, *, image):
    """Return generated token IDs for one image and a text question."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            image = opened_image.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a local image path or PIL.Image.Image")

    image_marker = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_marker = DEFAULT_IM_START_TOKEN + image_marker + DEFAULT_IM_END_TOKEN
    question = prompt if DEFAULT_IMAGE_TOKEN in prompt else image_marker + "\n" + prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer_image_token(
        formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    images = process_images([image], model._polestar_image_processor, model.config)
    if isinstance(images, list):
        images = [item.to(device=model.device, dtype=model.dtype) for item in images]
    else:
        images = images.to(device=model.device, dtype=model.dtype)

    generation = dict(config.generation)
    generation.pop("dtype", None)
    generation["gen_length"] = generation.pop("max_new_tokens", 256)
    generation.setdefault("steps", 256)
    generation.setdefault("block_length", 32)
    generation.setdefault("temperature", 0.0)
    generation.setdefault("threshold", 0.9)
    generation.setdefault("threshold_early", 0.7)
    generation.setdefault("use_cluster", True)
    generation.setdefault("tokenizer", tokenizer)
    with torch.inference_mode():
        return model.generate(
            input_ids,
            images=images,
            image_sizes=[image.size],
            **generation,
        )
