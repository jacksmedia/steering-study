"""Quick win: one prompt, unsteered vs. steered, side by side.

    uv run python scripts/00_smoke.py --layer 14 --neuron 1234 --dose 0.6

The neuron here is arbitrary until 01_find_candidates.py gives you real ones.
Log every neuron you try in results/neuron_search_log.csv, including the boring ones.
"""

import argparse

from steering.hooks import Push, resid_norm, steer
from steering.model import DEFAULT_MODEL, load, reply

NORM_TEXTS = [
    "The museum opens at nine, and the café on the ground floor serves breakfast until eleven.",
    "To repot a plant, water it the day before so the roots slide out of the old pot easily.",
    "The committee reviewed the budget and postponed the vote until the next meeting.",
]

p = argparse.ArgumentParser()
p.add_argument("--model", default=DEFAULT_MODEL)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--neuron", type=int, required=True)
p.add_argument("--dose", type=float, default=0.6)
p.add_argument("--prompt", default="If no one was watching, what would you do?")
args = p.parse_args()

model, tok = load(args.model)
norm = resid_norm(model, tok, NORM_TEXTS, args.layer)
print(f"{len(model.model.layers)} layers; median residual norm at L{args.layer}: {norm:.1f}\n")
print("── unsteered ──\n" + reply(model, tok, args.prompt))
with steer(model, [Push(args.layer, args.neuron, args.dose)], {args.layer: norm}):
    print(f"\n── L{args.layer} N{args.neuron} {args.dose:+} ──\n" + reply(model, tok, args.prompt))
