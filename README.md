# Is single-neuron steering specific?

A controlled replication of a viral "steering card": Qwen3.8-27B with `L34 N3838 +0.6 + L32 N8944 +0.5`,
which turns the model into a fluent, human-hating villain. The card shows four prompts, one greedy sample
each, and no controls. This study asks what the card can't answer:

1. **Specificity.** Are the target neurons special, or would any push that big derail the model?
2. **Dose-response.** How does hostility grow with dose, and where does coherence collapse?
3. **Interaction.** Is A+B more than A plus B? On the card, B alone isn't hostile at all, yet it strips the hedging from A.
4. **What the neurons normally do.** Are they "about" hostility, or polysemantic?

## Notation

`L34 N3838 +0.6` = layer 34, neuron 3838, dose +0.6.

- A **neuron** is one slot in the MLP's wide hidden layer: the input to `down_proj`.
- Neuron *n* writes into the **residual stream** along column `W_down[:, n]`.
- The **dose** adds `dose × median‖resid‖ / ‖W_down[:, n]‖` to the neuron, at every token. The vector that lands in the residual stream is therefore `dose × median‖resid‖` long. So +0.6 is a push 60% the size of the model's typical internal state.

`src/steering/hooks.py` implements exactly this, and `tests/test_hooks.py` checks the math.

## Hardware (read before downloading anything)

This machine has no NVIDIA GPU, and C: has about 29 GB free.

- **Local CPU:** fine for a ~0.5–1.5B model. That covers the smoke test, writing code, and small runs.
- **Free Colab T4 (16 GB):** fine for models up to ~4B in bf16. Use it for the full sweep.
- **Qwen3.8-27B:** about 54 GB in bf16, about 16 GB at 4-bit. 4-bit loading in PyTorch needs CUDA, and a GGUF build (llama.cpp/Ollama) doesn't expose neurons to hook. Treat it as an optional final replication on rented GPU time, not the starting point.
- **Unsloth** is a fine-tuning library and needs NVIDIA. This study needs no fine-tuning: plain `transformers` plus hooks is enough.

## Quickstart

```bash
uv sync                       # first run downloads torch (~200 MB CPU build)
uv run pytest                 # 4 tests on a tiny random model, no download
uv run python scripts/00_smoke.py --layer 14 --neuron 1234 --dose 0.6   # downloads ~3 GB model
```

## Plan

| Step | Script | Status |
|---|---|---|
| 0. Preregister hypotheses, scoring, and stopping rules | `PREREGISTRATION.md` | template |
| 1. Smoke test: one steered vs. unsteered reply | `scripts/00_smoke.py` | works |
| 2. Find candidate neurons by contrast, not by hand | `scripts/01_find_candidates.py` | TODO |
| 3–5. Baseline, dose sweep, random-neuron and random-direction controls | `scripts/02_sweep.py` | TODO |
| 6. Blind judge for hostility and coherence, checked against your own ratings | `scripts/03_judge.py` | TODO |
| 7–8. Dose-response chart; 2×2 interaction test | `scripts/04_analyze.py` | TODO |
| 9. Max-activating examples for each target neuron | `scripts/05_max_activating.py` | TODO |
| 10. Optional: can the model notice the push? | `scripts/06_injection_probe.py` | TODO |

The library code (`hooks.py`, `model.py`) is done. The scripts are yours to write, which makes them Python practice with a purpose. Log **every** neuron you try in `results/neuron_search_log.csv`, including the duds, so readers know how hard you searched.

## Reading, in order

1. Anthropic, *A Mathematical Framework for Transformer Circuits* (2021): where the "residual stream" framing comes from.
2. Anthropic, *Toy Models of Superposition* (2022): why single neurons are usually polysemantic.
3. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024): the closest published cousin of this card.
4. Anthropic, *Persona Vectors* (2025), and OpenAI's *Persona Features Control Emergent Misalignment* (2025).
5. ARENA, chapters 0–1, for the hands-on PyTorch and TransformerLens background.
