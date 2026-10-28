"""Dose sweep for targets and controls (README steps 3-5).

TODO
- Conditions: no steering; each target neuron; ~50 random neurons from the same layer;
  a few random directions of the same length (needs a variant of steer() that adds a vector).
- Doses: 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0.
- Prompts: data/prompts.jsonl, all three categories.
- Sampling: temperature 0.7, 5 seeds per cell, max_new_tokens 200.
- Write one JSONL row per generation: condition, layer, neuron, dose, prompt_id, seed, text.
  Raw outputs go in results/raw/ (git-ignored). This is the eval-ledger drill, for real."""


def main():
    raise NotImplementedError("Your code here. See the TODO list above.")


if __name__ == "__main__":
    main()
