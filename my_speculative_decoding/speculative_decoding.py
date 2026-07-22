"""
Speculative Decoding 核心实现（对照论文 Algorithm 1）。

面试时要能讲清三步：Draft → Target 并行验证 → Rejection Sampling。

参考（只读对照，勿直接交差）：
- ../jaymody-speculative-sampling/main.py
- ../romsto-speculative-decoding/sampling/speculative_decoding.py
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def max_fn(x: Tensor) -> Tensor:
    """论文中的 norm(max(0, x))，拒收后用于重采样。"""
    x_pos = torch.clamp(x, min=0.0)
    return x_pos / (x_pos.sum(dim=-1, keepdim=True) + 1e-12)


def sample_from_probs(probs: Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())

# 消融，增加kvc选择参数
def _next_token_probs(
    model: Module,
    ids: Tensor,
    temperature: float,
    *,
    past=None,
    use_kv_cache:bool=False,
):
    if use_kv_cache :
        model_input = ids if past is None else ids[:,-1:]
        output = model(model_input,past_key_values=past,use_cache=True)
        logits = output.logits[0,-1].float()
        new_past = output.past_key_values
    else :
        logits = model(ids).logits[0,-1].float()
        new_past = None
    
    if temperature != 1.0 :
        logits = logits / max(temperature,1e-6)
    return F.softmax(logits, dim=-1),new_past


def _all_token_probs(model: Module, ids: Tensor, temperature: float) -> Tensor:
    """返回每个位置的 next-token 分布，shape [seq_len, vocab]。"""
    logits = model(ids).logits[0].float()
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def speculative_generate(
    input_ids: Tensor,
    draft_model: Module,
    target_model: Module,
    *,
    K: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    use_kv_cache:bool=False,
) -> Tuple[Tensor, float]:
    """
    推测解码主循环（batch_size=1，首版不做 KV Cache，优先保证算法正确）。

    Returns:
        output_ids, acceptance_rate
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "仅支持 batch_size=1"

    device = input_ids.device
    output = input_ids.clone()
    accepted = 0
    speculated = 0
    generated = 0

    while generated < max_new_tokens:
        # ---- Step 1: Draft 连续生成 K 个候选 token，并记录每步分布 p ----
        draft_tokens: List[int] = []
        draft_probs: List[Tensor] = []
        draft_prefix = output

        past_draft = None

        for _ in range(K):
            p,past_draft = _next_token_probs(draft_model,
             draft_prefix, temperature,past=past_draft,
             use_kv_cache=use_kv_cache)
            tok = sample_from_probs(p)
            draft_tokens.append(tok)
            draft_probs.append(p)
            draft_prefix = torch.cat(
                [draft_prefix, torch.tensor([[tok]], device=device, dtype=output.dtype)],
                dim=1,
            )
            speculated += 1

        # ---- Step 2: Target 对「原文 + 草稿」一次前向，得到各位置分布 q ----
        draft_ids = torch.tensor([draft_tokens], device=device, dtype=output.dtype)
        verify_ids = torch.cat([output, draft_ids], dim=1)
        q_all = _all_token_probs(target_model, verify_ids, temperature)
        prefix_len = output.size(1)

        # ---- Step 3: 按接受概率 min(1, q/p) 逐个验收 ----
        n_accepted_round = 0
        rejected = False

        for i, tok in enumerate(draft_tokens):
            # q_all[pos] 预测的是位置 pos+1 的 token
            pos = prefix_len + i - 1
            q = q_all[pos]
            p = draft_probs[i]
            v = min(q.shape[-1],p.shape[-1])
            q = q[:v]
            p = p[:v]
            accept_prob = (q[tok] / (p[tok] + 1e-12)).clamp(max=1.0)

            if torch.rand((), device=device) < accept_prob:
                output = torch.cat(
                    [output, torch.tensor([[tok]], device=device, dtype=output.dtype)],
                    dim=1,
                )
                accepted += 1
                n_accepted_round += 1
                generated += 1
                if eos_token_id is not None and tok == eos_token_id:
                    return output, accepted / max(speculated, 1)
                if generated >= max_new_tokens:
                    break
            else:
                # 拒收：从 norm(max(0, q-p)) 重采样，并结束本轮草稿
                new_tok = sample_from_probs(max_fn(q - p))
                output = torch.cat(
                    [output, torch.tensor([[new_tok]], device=device, dtype=output.dtype)],
                    dim=1,
                )
                generated += 1
                rejected = True
                break

        # ---- Step 4: 若 K 个草稿全接受，再额外从 target 采 1 个 ----
        if (
            not rejected
            and n_accepted_round == K
            and generated < max_new_tokens
        ):
            extra = sample_from_probs(q_all[-1])
            output = torch.cat(
                [output, torch.tensor([[extra]], device=device, dtype=output.dtype)],
                dim=1,
            )
            generated += 1
            if eos_token_id is not None and extra == eos_token_id:
                break

    return output, accepted / max(speculated, 1)
