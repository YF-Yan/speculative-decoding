"""
端到端试跑：加载同系列小模型，对比自回归 vs 推测解码。

用法（在本目录下）:
  python benchmark.py
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import autoregressive_generate
from speculative_decoding import speculative_generate

# 默认用 GPT-2 家族，下载小、词表一致，适合先跑通
DRAFT_NAME = "gpt2"
TARGET_NAME = "gpt2-medium"
PROMPT = "Speculative decoding accelerates large language models by"
MAX_NEW_TOKENS = 48
K = 5
TEMPERATURE = 0.7


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"draft={DRAFT_NAME}  target={TARGET_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft = AutoModelForCausalLM.from_pretrained(DRAFT_NAME).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(TARGET_NAME).to(device).eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    # warmup
    _ = autoregressive_generate(input_ids, target, max_new_tokens=4, temperature=TEMPERATURE)

    t0 = time.perf_counter()
    ar_out = autoregressive_generate(
        input_ids,
        target,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        eos_token_id=tokenizer.eos_token_id,
    )
    t_ar = time.perf_counter() - t0

    t0 = time.perf_counter()
    sd_out, acc = speculative_generate(
        input_ids,
        draft,
        target,
        K=K,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        eos_token_id=tokenizer.eos_token_id,
    )
    t_sd = time.perf_counter() - t0

    ar_new = ar_out.size(1) - input_ids.size(1)
    sd_new = sd_out.size(1) - input_ids.size(1)

    print("\n=== Autoregressive (target only) ===")
    print(tokenizer.decode(ar_out[0], skip_special_tokens=True))
    print(f"time={t_ar:.3f}s  new_tokens={ar_new}  tokens/s={ar_new / t_ar:.2f}")

    print("\n=== Speculative Decoding ===")
    print(tokenizer.decode(sd_out[0], skip_special_tokens=True))
    print(f"time={t_sd:.3f}s  new_tokens={sd_new}  tokens/s={sd_new / t_sd:.2f}")
    print(f"acceptance_rate={acc:.3f}  speedup≈{(t_ar / t_sd):.2f}x (wall-clock)")


if __name__ == "__main__":
    main()
