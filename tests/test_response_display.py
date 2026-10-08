"""Human-facing response formatting must preserve the generated sequence."""

import sys
from types import SimpleNamespace

import torch

from polestar import PolestarConfig, generate
from polestar.inference import _decode_response


class Tokenizer:
    eos_token = "<|endoftext|>"
    eos_token_id = 3
    mask_token = None
    all_special_tokens = ["<|startoftext|>", "<|endoftext|>", "<|mdm_mask|>"]
    tokens = {
        1: "The answer is 42.",
        2: "<|eot_id|>",
        3: "<|endoftext|>",
        4: "<|mdm_mask|>",
        5: "This must stay after EOS.",
        6: "<|startoftext|>",
        7: " marks the end of a turn.",
        8: "The literal marker is <|eot_id|>",
        9: " \n",
        126336: "<|mdm_mask|>",
    }

    def decode(self, tokens, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(self.tokens[token] for token in tokens)

    def convert_ids_to_tokens(self, token_id):
        return self.tokens[token_id]

    def convert_tokens_to_ids(self, token):
        return {"<|eot_id|>": 2}[token]

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text):
        return {"input_ids": [99]}


def test_display_preserves_eos_cutoff_and_trims_terminal_chat_marker():
    token_ids = [6, 1, 2, 3, 5]
    assert _decode_response(Tokenizer(), token_ids, mask_token_id=126336, trim_chat_end=True) == "The answer is 42."
    assert token_ids == [6, 1, 2, 3, 5]


def test_interior_chat_marker_and_unresolved_mask_remain_visible():
    text = _decode_response(Tokenizer(), [2, 7, 4, 2], mask_token_id=126336, trim_chat_end=True)
    assert text == "<|eot_id|> marks the end of a turn.<|mdm_mask|>"


def test_other_models_do_not_trim_llada_chat_marker():
    assert _decode_response(Tokenizer(), [1, 2], mask_token_id=126336, trim_chat_end=False).endswith("<|eot_id|>")


def test_literal_marker_text_is_not_treated_as_chat_end_token():
    assert _decode_response(Tokenizer(), [8], mask_token_id=126336, trim_chat_end=True) == "The literal marker is <|eot_id|>"


def test_terminal_chat_marker_can_have_trailing_whitespace():
    assert _decode_response(Tokenizer(), [1, 2, 9, 3], mask_token_id=126336, trim_chat_end=True) == "The answer is 42."


def test_shared_generation_preserves_token_ids_nfe_and_rng(monkeypatch):
    sequences = torch.tensor([[99, 1, 2, 3, 5]])
    monkeypatch.setitem(
        sys.modules, "polestar.models.llada.generation",
        SimpleNamespace(generate_with_dual_cache=lambda *args, **kwargs: (sequences, 7, [], 1)),
    )
    rng_before = torch.get_rng_state().clone()
    result = generate(SimpleNamespace(device="cpu"), Tokenizer(), "Question", PolestarConfig("llada", "checkpoint"))
    assert result.text == "The answer is 42."
    assert torch.equal(result.token_ids, sequences[:, 1:])
    assert result.nfe == 7
    assert torch.equal(torch.get_rng_state(), rng_before)
