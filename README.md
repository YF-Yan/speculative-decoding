# Speculative Decoding 复现工作区

面向保研面试的论文复现：*Fast Inference from Transformers via Speculative Decoding* (ICML 2023)。

## 目录说明

| 目录 | 用途 |
|------|------|
| `romsto-speculative-decoding/` | **主参考实现**（只读对照，不要直接当自己的项目交） |
| `jaymody-speculative-sampling/` | **公式极简版**（对照论文 Algorithm 1，抠懂拒收采样） |
| `my_speculative_decoding/` | **你的正式复现**（手写核心 + 消融实验 + 面试材料） |

## 推荐学习顺序

1. 读论文 PDF（上一级目录）重点看 Algorithm 1  
2. 精读 `jaymody-speculative-sampling/main.py` 里的 `speculative_sampling`  
3. 对照 `romsto-speculative-decoding/sampling/speculative_decoding.py`  
4. 在 `my_speculative_decoding/` 里自己实现核心逻辑并做实验  

## 面试时怎么说

> 模型与 Tokenizer 用 HuggingFace 加载；草稿-验证循环与 Rejection Sampling 按论文公式用 PyTorch 手写；并完成不同草稿长度 K、温度、任务场景的消融实验。
