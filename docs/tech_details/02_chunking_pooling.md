# Chunking & Pooling（流式分块与语义池化）

## 0. 结论先行

- **核心作用**：将无限连续实时流按时间窗口或语义边界切分为宏块（Chunk），在宏块粒度上做双向建模，解决"流无终点 vs 双向 Encoder 需要完整序列"的根本矛盾。
- **工程推荐配置**：Tri-Transformer 目标 chunk_size=100ms（约 1–2 个语义 Token），注意力池化（Attention Pooling）压缩为单个全局表征，在延迟与语义质量间取得最优平衡。
- **轻量替代方案**：StreamingLLM（Sink Token + 滑动窗口）是 Chunking 的零配置替代，适合对架构改动极为保守的场景；Mamba/RWKV 等线性 RNN 以固定隐状态替代 KV Cache，是另一条路线。
- **Tri-Transformer 中的角色**：I-Transformer 第一阶（因果流式）与第二阶（双向 Encoder）的桥接层，控制每个宏块的语义粒度与延迟上限。

---

## 1. 概述

在处理无限连续实时流时，Transformer 面临两个根本矛盾：
1. **双向 Encoder 需要完整序列**，但实时流没有终点。
2. **全序列注意力**复杂度 $O(T^2)$，随流长度爆炸增长。

Chunking（分块）与 Pooling（池化）是解决这一矛盾的核心策略：将连续 Token 流按**时间窗口**或**语义边界**切分为宏块（Chunk），在宏块层面进行双向建模，获得压缩的语义表征。

**在 Tri-Transformer 中的角色**：I-Transformer 的 Chunking/Pooling 模块，将 Streaming Decoder 的细粒度因果输出聚合为 `i_enc`，供 C-Transformer 消费。

---

## 2. 实现原理

### 2.1 固定窗口分块（Fixed-Window Chunking）

每隔固定步数 $C$（如 100ms 对应 ~100 个 50ms 帧的 Token）切出一个 Chunk：

```python
class FixedWindowChunker:
    def __init__(self, chunk_size: int, d_model: int):
        self.chunk_size = chunk_size
        self.buffer = []
        self.pool = torch.nn.Linear(chunk_size * d_model, d_model)

    def push(self, token_embedding: torch.Tensor):
        """token_embedding: [d_model]"""
        self.buffer.append(token_embedding)
        if len(self.buffer) >= self.chunk_size:
            chunk = torch.stack(self.buffer, dim=0)
            self.buffer = []
            return self.pool(chunk.flatten())
        return None
```

### 2.2 注意力池化（Attention Pooling）

比简单平均更强的语义聚合，用可学习的 Query 向量提取 Chunk 内最重要的特征。本项目基于 Qwen3 架构，将 `MultiheadAttention` 替换为 GQA + QK-Norm + RoPE 的 `Qwen3Attention`：

```python
from app.model.branches import Qwen3Attention, Qwen3RMSNorm
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, num_kv_heads: int = 2):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = Qwen3Attention(
            hidden_size=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_qk_norm=True,
        )
        self.norm = Qwen3RMSNorm(d_model)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [batch, chunk_size, d_model]
        return: [batch, d_model]
        """
        q = self.query.expand(chunk.size(0), -1, -1)
        x = torch.cat([q, chunk], dim=1)
        x = self.norm(self.attn(x)[0])
        return x[:, 0]
```

### 2.3 语义边界感知分块（Semantic Boundary Chunking）

在音频流中，根据语音活动检测（VAD）或停顿位置切分，而非固定时间：

```python
class SemanticChunker:
    """基于 VAD 信号的语义边界分块"""
    def __init__(self, min_chunk=50, max_chunk=200):
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.buffer = []

    def push(self, token, is_silence: bool):
        self.buffer.append(token)
        n = len(self.buffer)
        if (is_silence and n >= self.min_chunk) or n >= self.max_chunk:
            chunk = self.buffer.copy()
            self.buffer = []
            return chunk
        return None
```

### 2.4 Longformer 滑动窗口注意力

Longformer（Beltagy et al., 2020）提出本地窗口注意力：每个 Token 仅关注半径 $w$ 内的邻居，同时引入少量全局 Token。其局部注意力模式与 Chunking 思想一致，但更为连续：

```
位置: 0  1  2  3  4  5  6  7
  0 [1  1  0  0  0  0  0  0]  ← 窗口宽=2
  1 [1  1  1  0  0  0  0  0]
  2 [0  1  1  1  0  0  0  0]
  3 [0  0  1  1  1  0  0  0]
  ...
  G [1  1  1  1  1  1  1  1]  ← Global Token，关注所有位置
```

---

## 3. 池化策略对比

| 策略 | 原理 | 计算代价 | 语义保留 | 适用场景 |
|---|---|---|---|---|
| 平均池化（Mean Pooling） | 对 Chunk 内所有 Token 取均值 | 极低 | 低 | 快速原型 |
| 最大池化（Max Pooling） | 取每维度最大值 | 极低 | 中 | 特征突出性强的场景 |
| 注意力池化（Attn Pooling） | 可学习 Query 提取关键特征 | 低-中 | 高 | 本项目推荐 |
| CLS Token 池化 | 在 Chunk 头部插入 [CLS]，用其输出 | 中 | 高 | BERT 风格编码器 |
| 分层池化（Hierarchical） | 多级 Chunking，递归聚合 | 高 | 最高 | 超长文档/视频 |

---

## 4. 使用方法

### 4.1 集成到 I-Transformer 流程

本项目使用 `Qwen3BidirectionalEncoderLayer`（来自 `branches.py`）替代原始 `BidirectionalEncoderLayer`：

```python
from app.model.branches import Qwen3BidirectionalEncoderLayer, Qwen3RMSNorm
import torch
import torch.nn as nn

class ITransformerChunkLayer(nn.Module):
    def __init__(self, d_model, chunk_size, n_encoder_layers,
                 num_heads=16, num_kv_heads=8):
        super().__init__()
        self.chunk_size = chunk_size
        self.attn_pool = AttentionPooling(d_model, num_heads, num_kv_heads)
        self.encoder_layers = nn.ModuleList([
            Qwen3BidirectionalEncoderLayer(d_model, num_heads, num_kv_heads)
            for _ in range(n_encoder_layers)
        ])
        self.norm = Qwen3RMSNorm(d_model)
        self.token_buffer: list[torch.Tensor] = []

    def forward_stream(self, dec_out: torch.Tensor):
        """dec_out: [batch, 1, d_model]，逐 Token 调用"""
        self.token_buffer.append(dec_out)
        if len(self.token_buffer) >= self.chunk_size:
            chunk = torch.cat(self.token_buffer, dim=1)
            self.token_buffer = []
            pooled = self.attn_pool(chunk).unsqueeze(1)
            x = pooled
            for layer in self.encoder_layers:
                x = layer(x)
            return self.norm(x)
        return None
```

---

## 5. 最新进展（2024-2025）

### 5.1 MegaByte（分层 Token 化）
- 将序列分为 Patch，每个 Patch 内用小模型处理，跨 Patch 用全局模型，是多尺度 Chunking 的极致形态。

### 5.2 RWKV / Mamba 的替代方案
- 线性 RNN（RWKV、Mamba）提供了另一种流式处理范式：以固定大小的隐状态替代 KV Cache，避免显式 Chunking，代价是全局建模能力弱于注意力机制。

### 5.3 StreamingLLM（无限上下文推理）
- MIT 2023 研究，通过保留 "Sink Token"（注意力汇聚的起始 Token）+ 滑动窗口，实现无限长流式生成，是 Chunking 的轻量化替代。

### 5.4 在多模态流中的应用
- Moshi：每 12.5ms 音频帧（对应 1 个语义 Token）处理一次，等效于 chunk_size=1 的极细粒度流式处理。
- Tri-Transformer 目标：chunk_size=100ms 为宏块语义边界，在延迟与语义质量间取得平衡。

---

## 6. MegaByte 与 MEGALODON 深度解析

### 6.1 MegaByte：分层多尺度 Token 化

MegaByte（Yu et al., Meta 2023，arXiv:2305.07185）是多尺度 Chunking 的极致实现，核心思想是将序列分为 Patch（宏块），用两个模型分别建模不同粒度：

```
原始字节序列（Byte-level）：
  [b1 b2 b3 b4 | b5 b6 b7 b8 | b9 b10 b11 b12]
       Patch 0        Patch 1        Patch 2

全局模型（Global Model，大模型）：
  输入: [patch_emb_0, patch_emb_1, ...]  (序列长度 = 总长 / patch_size)
  功能: 跨 Patch 的长程依赖建模

局部模型（Local Model，小模型）：
  输入: 单个 Patch 的字节序列
  功能: Patch 内精细生成（自回归）
```

**与 Tri-Transformer Chunking 的对照**：

| 维度 | MegaByte | Tri-Transformer Chunking |
|---|---|---|
| 粒度单位 | 原始字节 Patch | 语义 Token 宏块（100ms 音频帧） |
| 全局建模 | 大型 Global Transformer | Bidirectional Encoder（I-Transformer 第二阶） |
| 局部建模 | 小型 Local Transformer | Causal Streaming Decoder（I-Transformer 第一阶） |
| 主要收益 | 直接处理字节，消除 BPE tokenizer | 流式处理与全局理解解耦，降低延迟 |

```python
class MegaByteInspiredChunker(nn.Module):
    """受 MegaByte 启发的分层 Chunk 处理器"""

    def __init__(self, d_model: int, patch_size: int, n_global_layers: int):
        super().__init__()
        self.patch_size = patch_size
        self.local_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.patch_proj = nn.Linear(d_model * patch_size, d_model)
        self.global_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=n_global_layers,
        )

    def forward(self, token_stream: torch.Tensor) -> torch.Tensor:
        B, T, D = token_stream.shape
        assert T % self.patch_size == 0, "T 必须是 patch_size 的整数倍"

        local_out = self.local_encoder(token_stream)

        patches = local_out.view(B, T // self.patch_size, self.patch_size * D)
        patch_emb = self.patch_proj(patches)

        global_out = self.global_encoder(patch_emb)
        return global_out
```

### 6.2 MEGALODON：线性复杂度长上下文架构

MEGALODON（Ma et al., Meta 2024，arXiv:2404.08801）在 Mega（指数移动平均注意力）基础上进一步扩展，实现 $O(n)$ 复杂度的无限长上下文处理：

**核心组件**：

| 组件 | 作用 |
|---|---|
| CEMA（Chunk EMA） | 指数移动平均，在每个 Chunk 内并行计算，跨 Chunk 串行传递隐状态 |
| TimestepNorm | 时间步维度的归一化，替代传统 LayerNorm，适应流式场景 |
| GatedCrossAttention | 以 Chunk EMA 输出为 Key/Value，Query 来自当前 Token，降低注意力代价 |
| 多头门控注意力（MHGA） | 将 EMA 的全局偏置与局部 Token 注意力融合 |

**MEGALODON 与 Chunking 的关系**：MEGALODON 的 CEMA 机制本质上是可学习的 Chunking + EMA 池化，将每 Chunk 的信息压缩为固定大小的隐状态向量，是 Tri-Transformer `AttentionPooling` 的线性替代方案：

```python
class ChunkEMAPooling(nn.Module):
    """
    MEGALODON CEMA 启发的指数移动平均 Chunk 池化
    以固定大小隐状态替代注意力池化，O(1) 内存开销
    """

    def __init__(self, d_model: int, alpha_init: float = 0.9):
        super().__init__()
        self.d_model = d_model
        self.log_alpha = nn.Parameter(
            torch.full((d_model,), torch.log(torch.tensor(alpha_init)))
        )
        self.beta = nn.Parameter(torch.ones(d_model))
        self.gamma = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [batch, chunk_size, d_model]
        return: [batch, d_model]  (Chunk 的 EMA 隐状态)
        """
        alpha = torch.sigmoid(self.log_alpha)
        B, T, D = chunk.shape
        h = torch.zeros(B, D, device=chunk.device, dtype=chunk.dtype)
        for t in range(T):
            x_t = chunk[:, t, :]
            h = alpha * h + (1 - alpha) * (self.beta * x_t)
        return h * self.gamma

class MEGALODONInspiredEncoder(nn.Module):
    """Chunk EMA + Gated Attention 混合编码器"""

    def __init__(self, d_model: int, chunk_size: int, n_heads: int = 8):
        super().__init__()
        self.chunk_size = chunk_size
        self.ema_pool = ChunkEMAPooling(d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, token_stream: torch.Tensor) -> torch.Tensor:
        B, T, D = token_stream.shape
        n_chunks = T // self.chunk_size
        chunk_states = []

        for i in range(n_chunks):
            chunk = token_stream[:, i * self.chunk_size:(i + 1) * self.chunk_size, :]
            ema_state = self.ema_pool(chunk).unsqueeze(1)
            chunk_states.append(ema_state)

        ema_seq = torch.cat(chunk_states, dim=1)
        attn_out, _ = self.cross_attn(ema_seq, ema_seq, ema_seq)
        gate = torch.sigmoid(self.gate_proj(
            torch.cat([ema_seq, attn_out], dim=-1)
        ))
        return self.norm(gate * attn_out + (1 - gate) * ema_seq)
```

### 6.3 三种流式 Chunk 方案性能对比

| 方案 | 时间复杂度 | 空间复杂度 | 全局建模能力 | 延迟 | 适用场景 |
|---|---|---|---|---|---|
| **AttentionPooling**（本项目） | $O(C^2)$ per chunk | $O(C)$ | 高（Chunk 内双向注意力） | 低 | Tri-Transformer 推荐 |
| **MegaByte** 分层 | $O(C \log C)$ | $O(C)$ | 高（全局模型建模） | 中 | 字节级别长文档 |
| **ChunkEMA（MEGALODON）** | $O(C)$ | $O(1)$ | 中（EMA 记忆衰减） | 极低 | 超低延迟流式场景 |
| **StreamingLLM** 滑窗 | $O(W)$（W=窗口） | $O(W)$ | 低（无全局状态） | 极低 | 极简无改动部署 |
