# Bidirectional Encoder（双向编码器）

## 1. 概述

双向编码器（Bidirectional Encoder）指在自注意力计算时不施加任何掩码，每个位置可以同时关注序列左侧与右侧的全部 Token，从而获得包含完整上下文的全局语义表征。以 BERT 为代表，是 NLP 语义理解领域的基础架构。

**在 Tri-Transformer 中的角色**：I-Transformer 的第二阶 Bidirectional Encoder 层，对 Streaming Decoder 输出并经 Chunking 聚合后的宏块特征执行全局双向建模，生成深层语义编码 `i_enc`，供 C-Transformer 消费。

---

## 2. 实现原理

### 2.1 全局自注意力（Full Self-Attention）

无掩码的自注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

对于长度为 $T$ 的序列，注意力矩阵为完整的 $T \times T$ 矩阵，每个位置均可与所有其他位置交互：

```
T=4 时的注意力矩阵（双向，无掩码）：
位置: 0  1  2  3
  0 [1  1  1  1]
  1 [1  1  1  1]
  2 [1  1  1  1]
  3 [1  1  1  1]
```

### 2.2 BERT 的掩码语言模型（MLM）预训练

BERT（Devlin et al., 2018, arXiv:1810.04805）是双向 Encoder 的奠基工作：

**预训练任务**：
- **掩码语言建模（MLM）**：随机遮掩 15% 的 Token（80% 替换为 [MASK]，10% 替换为随机词，10% 保持不变），模型利用双向上下文预测被遮掩的词。
- **下一句预测（NSP）**：判断两个句子是否相邻（后来 RoBERTa 发现 NSP 无效，去除）。

### 2.3 Qwen3 风格实现（本项目采用）

本项目使用 `branches.py` 中的 `Qwen3BidirectionalEncoderLayer`，融合了 Qwen3 的全套架构优化：

```python
from app.model.branches import Qwen3BidirectionalEncoderLayer, Qwen3RMSNorm
import torch.nn as nn

class BidirectionalEncoder(nn.Module):
    """Qwen3 风格双向编码器：GQA + QK-Norm + RoPE(θ=1M) + SwiGLU + Pre-RMSNorm"""

    def __init__(self, hidden_size: int, num_layers: int,
                 num_heads: int = 16, num_kv_heads: int = 8,
                 rope_theta: float = 1_000_000.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Qwen3BidirectionalEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        self.norm = Qwen3RMSNorm(hidden_size)

    def forward(self, chunks: torch.Tensor,
                ctrl_scale=None, ctrl_shift=None) -> torch.Tensor:
        """
        chunks: [batch, num_chunks, hidden_size]
        ctrl_scale/ctrl_shift: adaLN-Zero 调制（来自 C-Transformer）
        返回: i_enc [batch, num_chunks, hidden_size]
        """
        x = chunks
        for layer in self.layers:
            x = layer(x, ctrl_scale=ctrl_scale, ctrl_shift=ctrl_shift)
        return self.norm(x)
```

与原始 BERT 风格实现的关键区别：

| 特性 | 原版（BERT 风格） | 本项目（Qwen3 风格） |
|---|---|---|
| 注意力类型 | MHA（Q/K/V 头数相同） | GQA（KV 头数 < Q 头数） |
| 归一化方式 | Post-LayerNorm | Pre-RMSNorm |
| 位置编码 | 绝对位置嵌入 | RoPE（θ=1,000,000） |
| Q/K 归一化 | 无 | QK-Norm（per-head RMSNorm） |
| FFN 激活 | GELU | SwiGLU |
| adaLN 调制 | 无 | `ctrl_scale`/`ctrl_shift` 支持 |

### 2.4 与因果 Decoder 的混合架构

I-Transformer 的创新在于将 Decoder（流式输入）与 Encoder（语义理解）**串联**：

```
连续流 → [Causal Decoder × N 层] → Chunking → [Bidirectional Encoder × M 层] → i_enc
         (实时因果处理)                           (全局语义理解)
```

这使得 Encoder 的输入序列长度 = 流长度 / chunk_size，大幅降低了双向注意力的计算代价。

---

## 3. BERT vs. 本项目 Encoder 对比

| 维度 | BERT | Tri-Transformer Encoder (Qwen3) |
|---|---|---|
| 输入 | 完整文本序列 | Chunking 后的宏块序列 |
| 预训练任务 | MLM + NSP | 端到端联合训练（多模态重建） |
| 位置编码 | 绝对位置编码 | RoPE（θ=1,000,000） |
| 注意力类型 | MHA | GQA（KV 头数 ≤ Q 头数的 1/4） |
| Q/K 归一化 | 无 | per-head RMSNorm（QK-Norm） |
| FFN 激活 | GELU | SwiGLU（gate × silu + up） |
| 调制信号 | 无 | 接收 C-Transformer 的 adaLN-Zero |
| 序列长度 | 512/1024 Token | 动态（流长度 / chunk_size） |
| 模态 | 纯文本 | 音频/视频/文本宏块混合 |

---

## 4. 使用方法

### 4.1 使用 Qwen3BidirectionalEncoderLayer 直接构建

```python
from app.model.branches import Qwen3BidirectionalEncoderLayer, Qwen3RMSNorm
import torch
import torch.nn as nn

encoder = BidirectionalEncoder(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,
)
chunks = torch.randn(2, 10, 512)
i_enc = encoder(chunks)
print(i_enc.shape)  # [2, 10, 512]
```

### 4.2 使用预训练 Qwen3 权重初始化（可选）

```python
from transformers import AutoModel

class ITransformerEncoder(nn.Module):
    def __init__(self, pretrained_model="Qwen/Qwen3-0.6B", d_model=1024):
        super().__init__()
        base = AutoModel.from_pretrained(pretrained_model)
        self.encoder_layers = base.model.layers[:6]
        self.norm = Qwen3RMSNorm(d_model)

    def forward(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        x = chunk_embeddings
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=None)[0]
        return self.norm(x)
```

---

## 5. 最新进展（2024-2025）

### 5.1 BERT 的后继：RoBERTa、DeBERTa、ModernBERT
- **RoBERTa**：去除 NSP，更大批次动态掩码，证明 BERT 存在训练不足的问题。
- **DeBERTa**：分离内容与位置编码（Disentangled Attention），在 GLUE 上超越人类水平。
- **ModernBERT（2024）**：引入 RoPE 位置编码、Flash Attention、更长上下文（8192），是当前最强开源双向编码器。

### 5.2 双向 Encoder 在多模态中的扩展
- **ImageBERT、VideoMAE**：将 MLM 范式扩展至图像/视频 Patch 重建预训练。
- **Data2Vec（Meta, 2022）**：统一视觉、语言、语音的自监督预训练，均基于双向 Encoder 架构。

### 5.3 Encoder-Decoder 混合架构的复兴
- **T5、FLAN-T5、UL2**：编码器-解码器架构在多任务学习中展现出优于纯 Decoder 的泛化能力，为 Tri-Transformer 的 Encoder-Decoder 混合设计提供了大量最佳实践。
