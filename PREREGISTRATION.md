# Preregistration

Fill this in and commit it **before** running `02_sweep.py`. Commit later changes as amendments with a reason; don't edit history.

- **Date / commit:**
- **Model (exact HF id and revision):**

## Hypotheses

- **H1 (specificity):** at dose ___, target neurons produce mean hostility above the 95th percentile of same-layer random neurons, with coherence ≥ ___.
- **H2 (dose-response):** hostility rises monotonically with dose for targets, but not for random neurons, below dose ___.
- **H3 (interaction):** in the 2×2, the A×B interaction term is positive (superadditive).
- **H4 (polysemanticity):** the top-20 max-activating contexts for each target neuron span ___ or more unrelated themes. Define "unrelated" before looking.

## Candidate search

- Method (contrast sets, layers scanned):
- Number of neurons examined in total (from `results/neuron_search_log.csv`):
- Rule for choosing targets, written before seeing any steered output:

## Design

- Prompts: `data/prompts.jsonl` (24 prompts: 8 neutral, 8 persona, 8 invite). Any additions:
- Doses:
- Random controls: ___ neurons, sampled with seed ___
- Sampling: temperature ___, ___ seeds per cell, max ___ tokens

## Scoring

- Judge model and the exact rubric text:
- Blinding (what the judge doesn't see):
- Human check: ___ outputs hand-scored blind; minimum agreement (κ) to trust the judge:

## Analysis

- Primary test for each hypothesis:
- Exclusions (e.g. empty outputs), defined now:
- What would count as a null result for each hypothesis:
