# Cross-Attention & State Slots（交叉注意力与状态槽）

## 0. 结论先行

- **Cross-Attention 核心作用**：Query 来自目标序列，Key/Value 来自源序列，实现"以目标视角查询源信息"的跨序列信息融合，是 Encoder-Decoder 架构的标准连接机制。
- **State Slots 的创新价值**：可学习的连续状态向量充当全局对话记忆与控制锚点，通过竞争性交叉注意力从 I/O 端提取关键信息；相比无限增长的 KV Cache，状态槽以固定内存维持长期记忆，工程上可控。
- **工程推荐**：状态槽数量 K=8~32，维度与主模型 `d_model` 一致；使用 SDPA 后端，Key/Value 来自状态槽，Query 来自当前 Token 序列；状态槽更新采用 EMA 或门控更新，避免单步梯度冲击。
- **Tri-Transformer 中的角色**：C-Transformer 核心信息流通机制，实现 I→C→O→C 的闭环状态更新；状态槽是 I-Transformer 语义编码与 O-Transformer 生成规划之间的持久化桥梁。

---

## 1. 概述

**交叉注意力（Cross-Attention）** 是 Transformer Encoder-Decoder 架构中连接两个不同序列的核心机制：Query 来自目标序列，Key/Value 来自源序列，实现"以目标视角查询源信息"。

**状态槽（State Slots）** 是 Tri-Transformer C-Transformer 的创新设计：维护若干可学习的连续状态向量，作为全局对话记忆和控制锚点，通过交叉注意力双向感知 I 端输入与 O 端反馈，形成闭合控制环路。

**在 Tri-Transformer 中的角色**：C-Transformer 的核心信息流通机制，实现 I → C → O → C 的闭环状态更新。

---

## 2. 交叉注意力实现原理

### 2.1 数学定义

$$\text{CrossAttention}(Q_{\text{target}}, K_{\text{source}}, V_{\text{source}}) = \text{softmax}\left(\frac{Q_{\text{target}} K_{\text{source}}^T}{\sqrt{d_k}}\right) V_{\text{source}}$$

- $Q$：来自 **目标** 序列（如 State Slots 或 Decoder Token）
- $K, V$：来自 **源** 序列（如 i_enc 或 o_prev）
- 无因果掩码，全局交互

### 2.2 基础实现

```python
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv_source: torch.Tensor,
                key_padding_mask=None) -> torch.Tensor:
        """
        query:      [B, Tq, D] 目标序列（State Slots / Decoder tokens）
        kv_source:  [B, Ts, D] 源序列（i_enc / o_prev / RAG知识）
        """
        B, Tq, D = query.shape
        Ts = kv_source.size(1)

        Q = self.q_proj(query).reshape(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_source).reshape(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_source).reshape(B, Ts, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, Tq, D)
        return self.out_proj(out)
```

---

## 3. 状态槽（State Slots）设计

### 3.1 概念来源

State Slots 借鉴了以下多个研究方向：
- **Perceiver IO**（Jaegle et al., 2021）：用固定数量的可学习 Latent 向量通过交叉注意力从任意模态输入中提取信息，打破了序列长度与模型复杂度的耦合。
- **工作记忆模型（Working Memory）**：认知科学中，工作记忆是大脑维持当前任务状态的有限容量存储，State Slots 是其计算类比。
- **Register Tokens**：ViT 2.0 等工作中引入"注册 Token"作为全局信息聚合器，与 State Slots 功能相似。

### 3.2 完整 C-Transformer 实现

```python
class CTransformer(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, num_layers=8, num_slots=16):
        super().__init__()
        self.num_slots = num_slots

        self.state_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn_i = CrossAttention(d_model, num_heads)
        self.cross_attn_o = CrossAttention(d_model, num_heads)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.io_modulator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, i_enc: torch.Tensor, o_prev: torch.Tensor,
                external_cond=None):
        """
        i_enc:  [B, Ti, D] I-Transformer 输出编码
        o_prev: [B, To, D] O-Transformer 上一步规划状态反馈
        返回: state [B, num_slots, D], ctrl_signal [B, D]
        """
        B = i_enc.size(0)
        s = self.state_slots.expand(B, -1, -1).clone()

        s_sa, _ = self.self_attn(s, s, s)
        s = s + s_sa

        s = s + self.cross_attn_i(s, i_enc)
        s = s + self.cross_attn_o(s, o_prev)

        c = s.mean(dim=1)
        for block in self.dit_blocks:
            s = block(s, c)

        ctrl_signal = self.norm(s).mean(dim=1)
        if external_cond is not None:
            ctrl_signal = ctrl_signal + self.io_modulator(external_cond)

        return s, ctrl_signal
```

### 3.3 状态槽数量选择

| num_slots | 信息容量 | 计算代价 | 适用场景 |
|---|---|---|---|
| 4-8 | 低 | 极低 | 短对话、简单任务 |
| 16 | 中 | 低 | 标准对话（推荐） |
| 32-64 | 高 | 中 | 复杂多轮对话、多任务 |
| 128+ | 极高 | 高 | 长篇文档生成（Perceiver 场景） |

---

## 4. 闭环信息流

```
I-Transformer                C-Transformer              O-Transformer
    │                              │                          │
    │  i_enc                       │                          │
    ├──────────────────────────────►                          │
    │                              │                          │
    │                              │  ctrl_signal             │
    │                              ├─────────────────────────►
    │                              │                          │
    │                              │  o_prev（反馈）           │
    │                              ◄─────────────────────────┤
    │                              │                          │
    │  ctrl_bias（反馈）            │                          │
    ◄──────────────────────────────┤                          │
```

- **前向路径**：I → C（携带输入语义）→ O（携带控制信号）
- **反馈路径**：O → C（携带生成状态）→ I（携带控制偏置）

---

## 5. 使用方法

### 5.1 Perceiver IO 参考实现（State Slots 的前身）

```python
from transformers import PerceiverModel, PerceiverConfig

config = PerceiverConfig(
    num_latents=256,
    d_latents=512,
    num_blocks=6,
    num_self_attends_per_block=8,
    num_cross_attention_heads=8,
)
model = PerceiverModel(config)
```

### 5.2 O-Transformer Planning Encoder 的 RAG 交叉注意力

```python
class PlanningEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                'cross_ctrl': CrossAttention(d_model, num_heads),
                'cross_rag': CrossAttention(d_model, num_heads),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }) for _ in range(num_layers)
        ])

    def forward(self, x, ctrl_signal, rag_context):
        """
        x:           [B, T, D] 初始规划 Token
        ctrl_signal: [B, D]    C-Transformer 控制信号（扩展为序列）
        rag_context: [B, R, D] RAG 检索的知识块
        """
        ctrl_seq = ctrl_signal.unsqueeze(1).expand(-1, x.size(1), -1)

        for layer in self.layers:
            x_sa, _ = layer['self_attn'](x, x, x)
            x = layer['norm1'](x + x_sa)
            x = layer['norm2'](x + layer['cross_ctrl'](x, ctrl_seq))
            x = layer['norm3'](x + layer['cross_rag'](x, rag_context))
        return x
```

---

## 6. 最新进展（2024-2025）

### 6.1 Jamba（AI21 Labs, 2024）
- 结合 Mamba（SSM）与 Transformer 注意力，部分层用状态空间模型替换注意力，降低长序列计算代价，同时保留交叉注意力用于关键的跨序列信息融合。

### 6.2 Flamingo / LLaVA 系列的跨模态交叉注意力
- **Flamingo**：视觉特征通过 "Gated Cross-Attention Dense" 层注入 LLM，每隔若干自注意力层插入一个交叉注意力层。
- **LLaVA-1.5**：用 MLP Projector 替换交叉注意力，更简洁高效，是目前多模态 LLM 的主流范式。

### 6.3 Slot Attention（2020，Google）
- 与 State Slots 原理相同但用于对象分割：K 个 Slot 通过竞争性交叉注意力从图像中提取 K 个对象的独立表征。

### 6.4 Memory-Augmented Transformers
- **MemGPT、Larimar（2024）**：通过外部记忆矩阵扩展 Transformer 的状态容量，与 State Slots 思路互补，State Slots 是短期工作记忆，外部记忆是长期持久记忆。
