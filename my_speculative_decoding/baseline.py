"""普通自回归生成，作为速度与分布对照的 baseline。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def autoregressive_generate(
    input_ids: Tensor,
    model: Module,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    output = input_ids.clone()
    device = input_ids.device

    for _ in range(max_new_tokens):
        logits = model(output).logits[0, -1].float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        tok = int(torch.multinomial(probs, num_samples=1).item())
        output = torch.cat(
            [output, torch.tensor([[tok]], device=device, dtype=output.dtype)],
            dim=1,
        )
        if eos_token_id is not None and tok == eos_token_id:
            break
    return output
