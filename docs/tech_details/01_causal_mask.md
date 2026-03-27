# Causal Mask（因果掩码）

## 1. 概述

因果掩码（Causal Mask），又称自回归掩码，是 Decoder-only Transformer 架构的核心机制。它通过在注意力矩阵上施加上三角遮掩，强制每个位置只能关注自身及其之前的 Token，从而实现时序因果性（Temporal Causality）和流式（Streaming）处理能力。

**在 Tri-Transformer 中的角色**：I-Transformer 的第一阶 Streaming Decoder 层使用因果掩码，实现 0 延迟的实时多模态流接收。

---

## 2. 实现原理

### 2.1 数学定义

标准自注意力（Self-Attention）计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

引入因果掩码后，注意力分数矩阵 $A = \frac{QK^T}{\sqrt{d_k}}$ 在 Softmax 前被施加掩码：

$$A_{ij} = \begin{cases} A_{ij} & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

其中 $i$ 为查询位置，$j$ 为键值位置。$-\infty$ 在 Softmax 后变为 0，即未来位置的权重为零。

### 2.2 掩码矩阵形式

对于序列长度 $T$，因果掩码为下三角矩阵：

```
T=4 时的掩码矩阵（1=允许关注，0=遮掩）：
位置: 0  1  2  3
  0 [1  0  0  0]
  1 [1  1  0  0]
  2 [1  1  1  0]
  3 [1  1  1  1]
```

### 2.3 PyTorch 实现

```python
import torch
import torch.nn.functional as F
import math

def causal_self_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    d_k = Q.size(-1)
    T = Q.size(-2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    causal_mask = torch.tril(torch.ones(T, T, device=Q.device)).bool()
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        Q, K, V = qkv.unbind(2)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        out = causal_self_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
```

### 2.4 KV Cache：流式推理加速

推理时，因果掩码允许增量计算（KV Cache 机制）：

```python
class StreamingDecoder:
    def __init__(self, model):
        self.model = model
        self.kv_cache = {}

    def step(self, new_token):
        """每次只处理 1 个新 Token，历史 KV 从缓存读取"""
        k_new, v_new = self.model.compute_kv(new_token)

        if 'k' in self.kv_cache:
            k_full = torch.cat([self.kv_cache['k'], k_new], dim=1)
            v_full = torch.cat([self.kv_cache['v'], v_new], dim=1)
        else:
            k_full, v_full = k_new, v_new

        self.kv_cache['k'] = k_full
        self.kv_cache['v'] = v_full

        q_new = self.model.compute_q(new_token)
        scores = torch.matmul(q_new, k_full.transpose(-2, -1)) / math.sqrt(q_new.size(-1))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v_full)
```

每步推理时间复杂度从 $O(T^2)$ 降为 $O(T)$（T 为当前序列长度）。

---

## 3. 与双向注意力的核心区别

| 特性 | 因果注意力（Causal） | 双向注意力（Bidirectional） |
|---|---|---|
| 上下文范围 | 仅左侧（历史） | 左右两侧（全局） |
| 掩码类型 | 下三角掩码 | 无掩码 |
| 流式处理 | 支持（逐 Token） | 不支持（需完整序列） |
| 延迟 | 0 延迟 | 需等序列结束 |
| 代表模型 | GPT、Llama、Qwen | BERT、RoBERTa、Encoder |
| 主要用途 | 生成（Decoder）、实时输入 | 理解（Encoder）、语义编码 |

---

## 4. 使用方法

### 4.1 在 Hugging Face Transformers 中使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

inputs = tokenizer("你好，", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        use_cache=True
    )

print(tokenizer.decode(outputs[0]))
```

### 4.2 流式生成（Streaming）

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True)
model.generate(**inputs, max_new_tokens=200, streamer=streamer)
```

---

## 5. 最新进展（2024-2025）

### 5.1 滑动窗口因果注意力（Sliding Window Causal Attention）
- **Mistral / Mixtral**：引入滑动窗口注意力（SWA），因果掩码仅覆盖最近 W 个 Token（如 W=4096），将长序列注意力复杂度从 $O(T^2)$ 降为 $O(T \cdot W)$。
- 适用于 Tri-Transformer I-Transformer 中处理超长连续流输入。

### 5.2 Grouped Query Attention（GQA）
- Llama 3、Qwen2 等均采用 GQA：多个查询头共享同一 KV 头，大幅减少 KV Cache 内存占用，适合长上下文流式推理。

### 5.3 RoPE（旋转位置编码）+ 长上下文扩展
- 因果注意力与 RoPE 结合，通过 YaRN、LongRoPE 等技术将上下文窗口扩展至 128K-1M Token，支持超长连续多模态流。

### 5.4 FlashAttention 加速因果计算
- FlashAttention-2/3 对因果掩码的下三角注意力做了专门的 CUDA 优化，跳过上三角无效计算，实际有效 FLOPs 减半，这是 Tri-Transformer 流式推理的关键加速依赖。

---

## 6. 与 Tri-Transformer 的关联

在 Tri-Transformer I-Transformer 中：
- **Streaming Decoder 层**：完全基于因果掩码自注意力，0 延迟接收多模态 Token 流。
- **KV Cache**：保存历史音视频 Token 的 K/V，每个新 Token 到来时仅计算增量注意力。
- **控制注入（Ctrl Bias）**：C-Transformer 的 adaLN-Zero 信号作为加性偏置注入，不破坏因果掩码的时序完整性。
