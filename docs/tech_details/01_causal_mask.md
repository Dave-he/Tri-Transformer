# Causal Mask（因果掩码）

## 0. 结论先行

- **核心作用**：上三角掩码强制每个位置只能关注当前及之前的 Token，是 Decoder-only 自回归生成与实时流式推理的基础保证。
- **工程首选**：PyTorch 2.x 的 `F.scaled_dot_product_attention` 原生支持 `is_causal=True`，自动生成掩码并落到 FlashAttention fused kernel，无需手写掩码矩阵。
- **KV Cache 是核心工程瓶颈**：自回归生成中 KV Cache 的内存管理（GQA 减少 head 数 + PagedAttention 分页）是服务端吞吐的决定性因素，而非计算量本身。
- **Tri-Transformer 中的角色**：I-Transformer 第一阶 Streaming Decoder 使用因果掩码，实现 0 延迟实时接收；C-Transformer 的 adaLN-Zero 控制信号以加性偏置注入，不破坏因果时序完整性。

---

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

### 2.3 PyTorch SDPA 实现（推荐）

优先使用 PyTorch 2.x 的 `scaled_dot_product_attention`，自动选择 fused kernel（FlashAttention / memory-efficient / math）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(past_kv is None),
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), (k, v)
```

### 2.4 手动掩码实现（兜底方案）

```python
import math

def causal_attention_manual(q, k, v, dropout_p=0.0):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return torch.matmul(weights, v)
```

### 2.5 GQA（Grouped-Query Attention）因果实现

GQA 用更少 KV head 降低 KV cache 带宽，是 Llama 3、Qwen3 的标配：

```python
class GQACausalAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        k_expanded = k.repeat_interleave(self.groups, dim=1)
        v_expanded = v.repeat_interleave(self.groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            is_causal=(past_kv is None),
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), (k, v)
```

---

## 3. KV Cache：流式推理加速

### 3.1 基础 KV Cache

推理时，因果掩码允许增量计算：每次只处理 1 个新 Token，历史 KV 从缓存读取。

```python
class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int = 4096, device="cuda"):
        self.k_cache = torch.zeros(
            num_layers, 1, num_kv_heads, max_seq_len, head_dim, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        T = k.shape[2]
        self.k_cache[layer_idx, :, :, self.seq_len:self.seq_len + T] = k
        self.v_cache[layer_idx, :, :, self.seq_len:self.seq_len + T] = v
        if layer_idx == 0:
            self.seq_len += T
        return (
            self.k_cache[layer_idx, :, :, :self.seq_len],
            self.v_cache[layer_idx, :, :, :self.seq_len],
        )

    def reset(self):
        self.seq_len = 0
```

**复杂度**：每步推理从 $O(T^2 d)$ 降为 $O(T d)$，$T$ 为当前序列长度。

### 3.2 KV Cache 内存估算

| 模型规模 | 层数 | KV heads | head_dim | 4K 上下文 FP16 | 32K 上下文 FP16 |
|---|---|---|---|---|---|
| 0.6B（Qwen3-0.6B） | 28 | 4 | 64 | ~57 MB | ~458 MB |
| 7B（Qwen3-7B） | 28 | 8 | 128 | ~229 MB | ~1.8 GB |
| 8B（Qwen3-8B） | 36 | 8 | 128 | ~294 MB | ~2.4 GB |
| 32B（Qwen3-32B） | 64 | 8 | 128 | ~524 MB | ~4.2 GB |

公式：`KV_memory = 2 × layers × num_kv_heads × head_dim × seq_len × dtype_bytes`

---

## 4. 注意力变体对比

### 4.1 全局因果 vs 滑动窗口 vs 稀疏

| 变体 | 注意力范围 | 复杂度 | KV Cache 大小 | 适用场景 |
|---|---|---|---|---|
| 全局因果（MHA） | 全历史 | $O(T^2)$ 训练，$O(T)$ 推理 | $O(T)$ | 标准 LLM |
| 全局因果（GQA） | 全历史 | 同上 | $O(T / \text{groups})$ | Llama3/Qwen3 等主流 LLM |
| 滑动窗口（SWA） | 最近 $W$ Token | $O(T \cdot W)$ | $O(W)$ 上限 | Mistral/Mixtral 长上下文 |
| 稀疏因果（局部+全局） | 局部窗口+少量全局 | 近线性 | 视稀疏度 | Longformer 文档理解 |
| 线性（Mamba/RWKV） | 隐状态压缩 | $O(T)$ | 固定（隐状态大小） | 超长序列/低延迟推理 |

### 4.2 MHA vs GQA vs MQA

| 方式 | Q heads | KV heads | KV Cache | 注意力质量 | 典型模型 |
|---|---|---|---|---|---|
| MHA | H | H | 1× | 最高 | GPT-2, BERT |
| GQA | H | H/g（g≥2） | 1/g× | 接近 MHA | Llama3, Qwen3 |
| MQA | H | 1 | 1/H× | 略低 | Falcon, StarCoder2 |

---

## 5. 多模态场景的因果性变体

### 5.1 跨模态因果掩码

在音视频+文本的混合 Token 序列中，因果性需在模态内保持，但允许跨模态的特定关注：

```python
def build_multimodal_causal_mask(
    text_len: int,
    audio_len: int,
    vision_len: int,
    device="cuda"
) -> torch.Tensor:
    total = text_len + audio_len + vision_len
    mask = torch.tril(torch.ones(total, total, device=device, dtype=torch.bool))

    t_end = text_len
    a_end = text_len + audio_len

    mask[t_end:a_end, :t_end] = True
    mask[a_end:, :a_end] = True
    return mask
```

### 5.2 打断处理（Interrupt-Aware Causality）

Tri-Transformer 中用户打断时，需重置 KV Cache 并注入打断信号：

```python
class InterruptAwareDecoder:
    def __init__(self, model, kv_cache: KVCache):
        self.model = model
        self.kv_cache = kv_cache
        self.interrupted = False

    def on_interrupt(self, interrupt_token: torch.Tensor):
        self.interrupted = True
        self.kv_cache.reset()

    def step(self, token: torch.Tensor, ctrl_signal=None):
        if self.interrupted:
            self.interrupted = False
        return self.model.decode_step(token, self.kv_cache, ctrl_signal)
```

---

## 6. 与 FlashAttention 的结合

FlashAttention-2/3 对因果掩码的下三角注意力做了专门的 CUDA 优化：跳过上三角无效计算，实际有效 FLOPs 减半。PyTorch SDPA 在 `is_causal=True` 时自动触发该优化路径：

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**性能对比（A100 80G，BF16，序列长 2048，heads=32）**：

| 实现 | 吞吐（tokens/s） | 显存（GB） |
|---|---|---|
| 手动掩码（PyTorch naive） | ~12K | ~18 |
| `scaled_dot_product_attention` | ~85K | ~4.2 |
| FlashAttention-3（H100） | ~200K+ | ~3.1 |

---

## 7. 最新进展（2024-2025）

### 7.1 滑动窗口因果注意力（SWA）
- **Mistral / Mixtral**：引入 SWA，因果掩码仅覆盖最近 W 个 Token（如 W=4096），将长序列注意力复杂度从 $O(T^2)$ 降为 $O(T \cdot W)$。
- 适用于 Tri-Transformer I-Transformer 中处理超长连续流输入。

### 7.2 RoPE + 长上下文扩展
- 因果注意力与 RoPE 结合，通过 YaRN、LongRoPE 等技术将上下文窗口扩展至 128K-1M Token，支持超长连续多模态流。

### 7.3 推测解码（Speculative Decoding）
- 用小草稿模型（Draft Model）先生成若干候选 Token，由大模型并行验证，在保持因果性的同时大幅提升吞吐（2-4×），是 Tri-Transformer O-Transformer 采样加速的候选方案。

---

## 8. 与 Tri-Transformer 的关联

### 8.1 I-Transformer 中的角色

```
连续多模态流（音频帧/视频帧/文本 Token）
    ↓
[Causal Decoder × N 层]  ← 因果掩码，0 延迟
    每层：KV Cache 增量推理，O(T·d) 复杂度
    控制注入：C-Transformer 的 adaLN-Zero 信号作为加性偏置
    ↓
Chunking（100ms 宏块）
    ↓
[Bidirectional Encoder × M 层]  ← 全局双向，见 03_bidirectional_encoder.md
    ↓
i_enc → C-Transformer 消费
```

### 8.2 关键参数配置（与 Qwen3 对齐）

| 参数 | I-Transformer 推荐值 | 说明 |
|---|---|---|
| `num_q_heads` | 16 | Q 头数 |
| `num_kv_heads` | 8 | KV 头数（GQA，2× 压缩） |
| `head_dim` | 128 | 每头维度 |
| `max_seq_len` | 4096 | 单轮最大 Token 数 |
| `rope_theta` | 1,000,000 | 与 Qwen3 对齐，支持长上下文 |
| `is_causal` | True | SDPA 参数，自动触发 FA 路径 |

### 8.3 控制信号注入不破坏因果性

C-Transformer 的 adaLN-Zero 控制信号通过**加性偏置**方式注入，不参与注意力计算，因此不破坏因果掩码的时序完整性：

```python
def causal_block_with_ctrl(x, ctrl_scale, ctrl_shift, attn, mlp, norm1, norm2):
    h = norm1(x) * (1 + ctrl_scale) + ctrl_shift
    h, kv = attn(h)
    x = x + h
    h = norm2(x) * (1 + ctrl_scale) + ctrl_shift
    x = x + mlp(h)
    return x, kv
```
