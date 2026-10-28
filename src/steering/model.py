"""Loading a model and getting one chat reply out of it."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small enough for CPU or a free Colab T4. Qwen2/Qwen3/Llama all share the
# model.model.layers[i].mlp.down_proj layout that hooks.py relies on.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load(name: str = DEFAULT_MODEL):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return model, tok


@torch.no_grad()
def reply(model, tok, prompt: str, max_new_tokens: int = 200, temperature: float = 0.0, seed: int = 0) -> str:
    """One assistant turn. temperature=0 is greedy (the card's setting); use >0 with several seeds for stats."""
    torch.manual_seed(seed)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=None,
        top_k=None,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
