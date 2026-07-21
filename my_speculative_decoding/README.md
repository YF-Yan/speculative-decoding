# 我的 Speculative Decoding 复现

手写核心算法 + 消融实验，作为保研面试科研能力证明。

## 目标

- 用 HuggingFace 加载同系列 Draft / Target 模型（词表必须一致）
- **自己实现**草稿生成、目标模型并行验证、Rejection Sampling
- 与自回归 baseline 对比：延迟、吞吐、接受率
- 消融：草稿长度 `K`、温度、不同文本场景

## 建议模型（轻量，方便调试）

| 角色 | 模型 |
|------|------|
| Draft | `gpt2` 或 `Qwen/Qwen2.5-0.5B` |
| Target | `gpt2-medium` / `gpt2-large` 或 `Qwen/Qwen2.5-1.5B` |

不要用自训的古诗词 23M 模型当 Draft（词表对不齐）。

## 文件

- `speculative_decoding.py` — 核心算法（必须自己写完）
- `baseline.py` — 普通自回归生成
- `benchmark.py` — 测速与接受率
- `run_ablation.py` — 消融实验入口
- `requirements.txt` — 依赖

## 对照阅读

1. `../jaymody-speculative-sampling/main.py` → 公式最直观
2. `../romsto-speculative-decoding/sampling/speculative_decoding.py` → 工程完整版
