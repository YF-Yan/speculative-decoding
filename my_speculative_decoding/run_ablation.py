"""
消融实验：扫描草稿长度 K，记录接受率与加速比。

用法:
  python run_ablation.py
结果会写入 ablation_results.csv，便于画图写进简历/报告。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import autoregressive_generate
from speculative_decoding import speculative_generate

DRAFT_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_NAME = "Qwen/Qwen2.5-7B"
PROMPTS = [
    "def fibonacci(n):",
    "Once upon a time in a quiet village,",
    "The capital of France is",
]
# K_LIST = [2, 3, 4, 5, 8] # k消融
K = 2
MAX_NEW_TOKENS = 40
# TEMPERATURE = 0.7
TEMP_LIST = [0.0,0.3, 0.5, 0.7,0.9,1] # 温度消融
OUT_CSV = Path("ablation_temp.csv")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    draft = AutoModelForCausalLM.from_pretrained(DRAFT_NAME,torch_dtype=dtype).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(TARGET_NAME,torch_dtype=dtype).to(device).eval()

    rows = []
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        for TEMPERATURE in TEMP_LIST:
            t0 = time.perf_counter()
            ar_out = autoregressive_generate(
                input_ids, target, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, use_kv_cache=True
            )
            t_ar = time.perf_counter() - t0
            ar_tps = (ar_out.size(1) - input_ids.size(1)) / t_ar


            t0 = time.perf_counter()
            sd_out, acc = speculative_generate(
                input_ids,
                draft,
                target,
                K=K,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                use_kv_cache=True,
            )
            t_sd = time.perf_counter() - t0
            sd_tps = (sd_out.size(1) - input_ids.size(1)) / t_sd
            rows.append(
                {
                    "prompt": prompt[:40],
                    "K": K,
                    "temperature": TEMPERATURE,
                    "acceptance_rate": round(acc, 4),
                    "ar_tokens_per_s": round(ar_tps, 3),
                    "sd_tokens_per_s": round(sd_tps, 3),
                    "speedup": round(t_ar / t_sd, 3),
                }
            )
            print(rows[-1])

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved -> {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
