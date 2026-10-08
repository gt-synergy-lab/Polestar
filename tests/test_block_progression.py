from types import SimpleNamespace

import pytest
import torch

from polestar.models.llada.generation import generate_with_dual_cache


class Predictor:
    device = torch.device("cpu")

    def __init__(self, suffix_token):
        self.calls = 0
        self.suffix_token = suffix_token

    def __call__(self, tokens, **kwargs):
        self.calls += 1
        batch, length = tokens.shape
        logits = torch.zeros(batch, length, 3)
        logits[..., 1] = 20
        if self.calls == 1:
            logits[:, 3, :] = 0
        indices = torch.tensor([4, 5, 6]) if self.calls == 2 else None
        selected_logits = None
        if indices is not None:
            selected_logits = torch.zeros(batch, 3, 3)
            selected_logits[..., self.suffix_token] = 20
        return SimpleNamespace(
            logits=logits,
            past_key_values=(),
            past_hidden_states=[],
            clustered_hidden_states=[],
            drift_scores=[torch.zeros(3)],
            logits_sel=selected_logits,
            sel_indices=indices,
        )


@pytest.mark.parametrize("suffix_token", [1, 2])
def test_completed_middle_block_does_not_skip_remaining_blocks(suffix_token):
    model = Predictor(suffix_token)
    tokens, nfe, stats, _ = generate_with_dual_cache(
        model, torch.tensor([[0]]), steps=3, gen_length=9, block_length=3,
        mask_id=9, threshold=0.9, threshold_early=0.7,
    )
    assert not (tokens[:, 1:] == 9).any()
    assert tokens[0, 4:7].tolist() == [suffix_token] * 3
    assert model.calls == nfe == 3
    assert [row["block_id"] for row in stats] == [0, 2]
    assert sum(row["block_nfe"] for row in stats) == nfe
