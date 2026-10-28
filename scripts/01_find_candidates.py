"""Find candidate neurons without hand-hunting (README step 2).

TODO
- Write data/contrast/hostile.txt and friendly.txt yourself: ~50 lines each, matched in topic
  and length, so the only systematic difference is the persona.
- For a few middle layers, use neuron_activations() to get each neuron's mean activation on
  each set (average over tokens, then over texts).
- Rank neurons by (hostile mean - friendly mean). Save the top 10 per layer to results/.
- Log the method and every neuron you carry forward in results/neuron_search_log.csv.
- Decide BEFORE looking at steered outputs which neurons count as "targets" (preregistration)."""


def main():
    raise NotImplementedError("Your code here. See the TODO list above.")


if __name__ == "__main__":
    main()
