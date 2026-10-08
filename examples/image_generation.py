"""Answer a question about a local image with Polestar and LLaDA-V."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Path to an image")
    parser.add_argument("--prompt", default="Please describe the image in detail.")
    parser.add_argument("--config", type=Path, default=Path("configs/llada-v.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    from polestar import PolestarConfig, generate, load_model

    config = PolestarConfig.from_json(args.config)
    model, tokenizer = load_model(config, device=args.device)
    result = generate(model, tokenizer, args.prompt, config, image=args.image)
    print(result.text)


if __name__ == "__main__":
    main()
