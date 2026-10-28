"""Checks the steering math on a tiny randomly initialised Qwen2 model (no download needed)."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from steering.hooks import Push, down_proj, neuron_activations, resid_norm, steer

LAYER, NEURON = 1, 7


class FakeTok:
    """Just enough tokenizer for the hooks: maps characters to ids."""

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[ord(c) % 100 for c in text]])
        return _Batch(input_ids=ids)


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg).eval()


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


IDS = torch.tensor([[1, 5, 9, 13, 2, 8]])


def test_zero_dose_changes_nothing(model):
    base = logits(model, IDS)
    with steer(model, [Push(LAYER, NEURON, 0.0)], {LAYER: 3.0}):
        assert torch.allclose(logits(model, IDS), base)


def test_hook_is_removed_afterwards(model):
    base = logits(model, IDS)
    with steer(model, [Push(LAYER, NEURON, 0.8)], {LAYER: 3.0}):
        assert not torch.allclose(logits(model, IDS), base)
    assert torch.allclose(logits(model, IDS), base)


def test_push_length_is_dose_times_resid_norm(model):
    """What lands in the residual stream should be exactly dose * norm along W_down[:, n]."""
    outs = []
    handle = down_proj(model, LAYER).register_forward_hook(lambda m, a, out: outs.append(out[0, 0]))
    logits(model, IDS[:, :1])
    with steer(model, [Push(LAYER, NEURON, 0.6)], {LAYER: 5.0}):
        logits(model, IDS[:, :1])
    handle.remove()
    added = outs[1] - outs[0]
    column = down_proj(model, LAYER).weight[:, NEURON]
    assert added.norm().item() == pytest.approx(0.6 * 5.0, rel=1e-4)
    assert torch.nn.functional.cosine_similarity(added, column, dim=0).item() == pytest.approx(1.0, abs=1e-4)


def test_capture_shapes_and_norm(model):
    tok = FakeTok()
    acts = neuron_activations(model, tok, ["hello there", "hi"], LAYER)
    assert [a.shape for a in acts] == [(11, 64), (2, 64)]
    assert resid_norm(model, tok, ["hello there", "hi"], LAYER) > 0
