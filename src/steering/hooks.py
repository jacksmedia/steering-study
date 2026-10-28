"""Single-neuron steering and activation capture for Llama/Qwen-style decoder models.

Vocabulary (matches the steering card's notation, e.g. "L34 N3838 +0.6"):
  - layer L:   index into model.model.layers
  - neuron n:  index into the MLP's wide hidden layer, i.e. the INPUT to mlp.down_proj
  - dose:      size of the push, as a fraction of the typical residual-stream norm at layer L

down_proj is an nn.Linear(intermediate, hidden), so its weight is [hidden, intermediate] and
column n, W_down[:, n], is the direction neuron n writes into the residual stream.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Push:
    layer: int
    neuron: int
    dose: float


def down_proj(model, layer: int) -> torch.nn.Linear:
    return model.model.layers[layer].mlp.down_proj


@torch.no_grad()
def resid_norm(model, tok, texts: list[str], layer: int) -> float:
    """Median L2 norm of the residual stream entering `layer`, over every token but the first.

    hidden_states[0] is the embedding output, so hidden_states[L] is the input to layer L.
    The first position is skipped because it often carries a huge "attention sink" norm.
    """
    norms = []
    for text in texts:
        ids = tok(text, return_tensors="pt").to(model.device)
        hs = model(**ids, output_hidden_states=True).hidden_states[layer]
        norms.append(hs[0, 1:].float().norm(dim=-1))
    return torch.cat(norms).median().item()


@contextmanager
def steer(model, pushes: list[Push], norms: dict[int, float]):
    """Add each push to its neuron at every token position while the block is active.

    The amount added to neuron n is dose * norms[layer] / ||W_down[:, n]||, so the vector
    that lands in the residual stream has length dose * norms[layer], the card's formula.
    """
    handles = []
    for p in pushes:
        proj = down_proj(model, p.layer)
        delta = p.dose * norms[p.layer] / proj.weight[:, p.neuron].float().norm().item()

        def hook(module, args, neuron=p.neuron, delta=delta):
            x = args[0].clone()
            x[..., neuron] += delta
            return (x,) + args[1:]

        handles.append(proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def neuron_activations(model, tok, texts: list[str], layer: int) -> list[torch.Tensor]:
    """Per-text [seq_len, intermediate] activations at the down_proj input of `layer`."""
    captured = []
    handle = down_proj(model, layer).register_forward_pre_hook(
        lambda module, args: captured.append(args[0][0].float().cpu())
    )
    try:
        for text in texts:
            model(**tok(text, return_tensors="pt").to(model.device))
    finally:
        handle.remove()
    return captured
