"""A shared entry point for Polestar text and image generation."""

from dataclasses import dataclass
from typing import Any

from .config import PolestarConfig


@dataclass
class GenerationResult:
    """Generated text, generated token IDs, and the number of model evaluations."""

    text: str
    token_ids: Any
    nfe: int | None


def load_model(config: PolestarConfig, device: str = "cuda") -> tuple[Any, Any]:
    """Load the Polestar model implementation and checkpoint tokenizer."""
    if config.model_type == "llada_v":
        from .models.llada_v import load_model as load_vision_model

        return load_vision_model(config, device=device)

    import torch
    from transformers import AutoConfig, AutoTokenizer

    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}

    if config.model_type == "llada":
        from .models.llada import LLaDAModelLM

        model_class = LLaDAModelLM
        model_config = AutoConfig.from_pretrained(config.model_id)
        model_config.flash_attention = True
        model_kwargs["config"] = model_config
    elif config.model_type == "dream":
        from .models.dream import DreamModel

        model_class = DreamModel
    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")

    model = model_class.from_pretrained(config.model_id, **model_kwargs).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: PolestarConfig,
    *,
    image: Any = None,
) -> GenerationResult:
    """Generate one response using the selected model's chat template."""
    if config.model_type == "llada_v":
        from .models.llada_v import generate as generate_vision

        if image is None:
            raise ValueError("LLaDA-V generation requires image=path or image=PIL.Image.Image")
        token_ids = generate_vision(model, tokenizer, prompt, config, image=image)
        nfe = getattr(model, "_fast_dllm_last_stats", {}).get("nfe")
    else:
        import torch

        if image is not None:
            raise ValueError("Image inputs require a LLaDA-V preset")
        settings = config.generation
        messages = [{"role": "user", "content": prompt}]
        length = settings.get("max_new_tokens", 256)
        steps = settings.get("steps", 256)
        block_length = settings.get("block_length", 32)
        temperature = settings.get("temperature", 0.0)
        threshold = settings.get("threshold", 0.9)
        threshold_early = settings.get("threshold_early", 0.7)

        if config.model_type == "llada":
            from .models.llada.generation import generate_with_dual_cache

            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = torch.tensor(tokenizer(text)["input_ids"], device=model.device).unsqueeze(0)
            sequences, nfe, _, _ = generate_with_dual_cache(
                model,
                input_ids,
                steps=steps,
                gen_length=length,
                block_length=block_length,
                temperature=temperature,
                remasking=settings.get("remasking", "low_confidence"),
                mask_id=settings.get("mask_id", 126336),
                threshold=threshold,
                threshold_early=threshold_early,
                use_cluster=True,
            )
            token_ids = sequences[:, input_ids.shape[1]:]
        elif config.model_type == "dream":
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            input_ids = inputs.input_ids.to(model.device)
            output = model.diffusion_generate(
                input_ids,
                attention_mask=inputs.attention_mask.to(model.device),
                max_new_tokens=length,
                steps=steps,
                temperature=temperature,
                top_p=settings.get("top_p"),
                top_k=settings.get("top_k"),
                block_length=block_length,
                threshold=threshold,
                threshold_early=threshold_early,
                alg="confidence_threshold",
                dual_cache=True,
                use_cluster=True,
                return_dict_in_generate=True,
                output_history=False,
            )
            token_ids = output.sequences[:, input_ids.shape[1]:]
            nfe = output.nfe
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

    text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=False)
    if tokenizer.eos_token:
        text = text.split(tokenizer.eos_token, 1)[0]
    return GenerationResult(text=text.strip(), token_ids=token_ids, nfe=nfe)
