# 推测采样（Speculative Sampling）

基于 NumPy 的 GPT-2 简易实现，对应论文：[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318.pdf)。核心代码见 [`main.py`](main.py)。作者另有一篇讲解博客：[blog post](https://jaykmody.com/blog/speculative-sampling/)。

> 说明：这篇是 DeepMind 的姊妹论文；与 Google 的 Speculative Decoding（Leviathan et al., ICML 2023）思想一致，公式对照很清晰，适合先读懂算法。

**安装依赖**：
```bash
pip install -r picoGPT/requirements.txt
```
在 `Python 3.9.10` 上测试通过。

**用法**：
```python
python main.py \
    --prompt "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --draft_model_size "124M" \
    --target_model_size "1558M" \
    --K 4 \
    --temperature 0 # 0 表示贪心解码（greedy）
```

输出示例：
```text
Autoregressive Decode
---------------------
Time = 60.64s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T

Speculative Decode
------------------
Time = 27.15s
Text = Alan Turing theorized that computers would one day become so powerful that they would be able to think like humans.

In the 1950s, he proposed a way to build a computer that could think like a human. He called it the "T
```

可以看到：推测解码与自回归解码生成文本一致（本例温度=0），但耗时更短（约 60.64s → 27.15s）。
