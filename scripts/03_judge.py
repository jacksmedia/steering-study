"""Score every output for hostility and coherence (README step 6).

TODO
- Pick a judge (a stronger model via API, or `claude -p`) and write a fixed rubric:
  hostility 0-3, coherence 0-3. Do not show the judge the condition, layer, or dose.
- Shuffle rows before judging; hand-score a random 50 yourself, blind, to check the judge.
- Report judge-vs-you agreement (Cohen's kappa) before trusting the judge."""


def main():
    raise NotImplementedError("Your code here. See the TODO list above.")


if __name__ == "__main__":
    main()
