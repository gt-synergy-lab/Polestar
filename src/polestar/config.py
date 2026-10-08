"""Model and generation settings for Polestar."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class PolestarConfig:
    """Load a checkpoint and its generation settings from a JSON preset."""

    model_type: str
    model_id: str
    generation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_type not in {"llada", "dream", "llada_v"}:
            raise ValueError("model_type must be llada, dream, or llada_v")
        if not self.model_id:
            raise ValueError("model_id must name a checkpoint or local checkpoint directory")
        if not isinstance(self.generation, dict):
            raise ValueError("generation must be a JSON object")
        length = self.generation.get("max_new_tokens", 256)
        block = self.generation.get("block_length", 32)
        steps = self.generation.get("steps", 256)
        if not all(isinstance(value, int) and value > 0 for value in (length, block, steps)):
            raise ValueError("max_new_tokens, block_length, and steps must be positive integers")
        if length % block:
            raise ValueError("max_new_tokens must be divisible by block_length")
        if steps % (length // block):
            raise ValueError("steps must be divisible by the number of generation blocks")

    @classmethod
    def from_json(cls, path: str | Path) -> "PolestarConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))
