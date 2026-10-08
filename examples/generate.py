"""Generate a response with a checked-in Polestar model preset."""

import argparse
import random

from polestar import PolestarConfig, generate, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with Polestar")
    parser.add_argument("--config", default="configs/llada8b.json", help="Model preset JSON")
    parser.add_argument("--prompt", default="Explain why the sky is blue in a few sentences.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, help="Override the preset generation length")
    parser.add_argument("--image", help="Image path for a LLaDA-V preset")
    args = parser.parse_args()

    config = PolestarConfig.from_json(args.config)
    if args.max_new_tokens is not None:
        config.generation["max_new_tokens"] = args.max_new_tokens
        config.__post_init__()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model(config, device=args.device)
    result = generate(model, tokenizer, args.prompt, config, image=args.image)
    print(result.text)
    if result.nfe is not None:
        print(f"\nModel evaluations: {result.nfe}")


if __name__ == "__main__":
    main()
