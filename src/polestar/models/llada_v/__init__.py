"""Image-conditioned Polestar inference with LLaDA-V."""


def load_model(config, device="cuda"):
    from .loader import load_model as _load_model

    return _load_model(config, device=device)


def generate(model, tokenizer, prompt, config, *, image):
    from .loader import generate as _generate

    return _generate(model, tokenizer, prompt, config, image=image)


__all__ = ["load_model", "generate"]
