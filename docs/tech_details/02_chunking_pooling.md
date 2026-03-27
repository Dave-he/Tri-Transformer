# Chunking & Pooling（流式分块与语义池化）

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

比简单平均更强的语义聚合，用可学习的 Query 向量提取 Chunk 内最重要的特征：

```python
class AttentionPooling(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = torch.nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: [batch, chunk_size, d_model]
        return: [batch, d_model]
        """
        q = self.query.expand(chunk.size(0), -1, -1)
        out, _ = self.attn(q, chunk, chunk)
        return out.squeeze(1)
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

```python
class ITransformerChunkLayer(torch.nn.Module):
    def __init__(self, d_model, chunk_size, n_encoder_layers):
        super().__init__()
        self.chunk_size = chunk_size
        self.streaming_decoder = CausalTransformerDecoder(d_model)
        self.attn_pool = AttentionPooling(d_model)
        self.bidirectional_encoder = BidirectionalEncoder(d_model, n_encoder_layers)
        self.token_buffer = []

    def forward_stream(self, token: torch.Tensor, ctrl_bias=None):
        """逐 Token 调用，积累 chunk_size 后触发 Encoder"""
        dec_out = self.streaming_decoder.step(token, ctrl_bias)
        self.token_buffer.append(dec_out)

        if len(self.token_buffer) >= self.chunk_size:
            chunk = torch.stack(self.token_buffer, dim=1)
            self.token_buffer = []
            pooled = self.attn_pool(chunk)
            i_enc = self.bidirectional_encoder(pooled.unsqueeze(1))
            return i_enc
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
