# Chameleon（早期融合混合模态模型）

## 0. 结论先行

- **核心创新**：图像 Token 与文本 Token 从嵌入层起完全混合，单一 Transformer 主干在统一 Token 空间处理任意序列，无需独立视觉分支或适配器，是早期融合路线的最激进代表。
- **训练稳定性方案**：QK-Norm（逐头归一化防止注意力 logits 爆炸）+ z-loss（防止 Softmax 饱和）是早期融合混合模态大规模稳定训练的关键技术，已被 Qwen3 等模型采用。
- **工程推荐**：早期融合需要完整的多模态预训练数据，开发资源紧张时建议先走晚期融合（LLaVA 风格）稳定对齐各模态后再探索早期融合路线；QK-Norm 可作为低成本稳定性增强项直接加入现有 Transformer Block。
- **Tri-Transformer 中的角色**：为统一嵌入空间的稳定训练提供技术参考；QK-Norm 技术被 C-Transformer 采用以提升多模态联合训练稳定性；统一 Token 空间的词表设计借鉴 Chameleon 的模态边界 Token 方案。

---

## 1. 概述

Chameleon 是 Meta AI 于 2024 年发布的早期融合（Early-Fusion）Token 化混合模态基础模型（arXiv:2405.09818）。其核心创新是：**图像 Token 与文本 Token 从嵌入层起完全混合处理**，单一 Transformer 主干在统一 Token 空间中建模任意序列的图文混合内容，无需独立的视觉分支或跨模态适配器。

Chameleon 是 Any-to-Any 范式中处理方式最激进的代表，提出了混合模态早期融合的稳定训练方法，与 AnyGPT 的"数据层统一"方法相互补充。

**在 Tri-Transformer 中的角色**：为多模态统一嵌入空间的**稳定训练**方案提供参考，尤其是解决不同模态 Token 分布差异导致训练崩溃的技术挑战。

---

## 2. 早期融合 vs. 晚期融合

### 2.1 三种融合范式

```
晚期融合（Late Fusion）：LLaVA-1.5 风格
图像 → [Visual Encoder] → Visual Features
                               ↓ Linear Projector
文本 → [Text Embedding] → ── + ──→ [LLM Backbone]

早期融合（Early Fusion）：Chameleon 风格
图像 → [Image Tokenizer/VQ-VAE] → Visual Tokens
文本 → [Text Tokenizer/BPE] → Text Tokens
                           ↓
              统一 Embedding 矩阵（共享权重）
                           ↓
              [标准 Transformer Decoder]（直接混合处理）

超早期融合（Pixel-level）：PixelBERT 风格
图像像素 + 文本直接拼接，计算成本极高，实用性低
```

### 2.2 早期融合的优劣

| 维度 | 早期融合（Chameleon） | 晚期融合（LLaVA） |
|---|---|---|
| 统一性 | 最高（单一Embedding空间） | 中（需适配器对齐） |
| 扩展性 | 高（易添加新模态） | 中 |
| 训练稳定性 | 难（分布不匹配问题） | 容易 |
| 图生文理解 | 强 | 强 |
| 文生图生成 | 支持（端到端） | 不支持（需外接扩散模型）|
| 推理效率 | 高（单一模型） | 中 |

---

## 3. 架构与训练稳定化

### 3.1 图像 Tokenizer

Chameleon 使用定制的 VQ-VAE，将 512×512 图像编码为 1024 个离散 Token（32×32），码本大小 8192：

```python
class ChameleonImageTokenizer:
    IMAGE_VOCAB_SIZE = 8192
    TOKENS_PER_IMAGE = 1024

    def encode(self, image: PIL.Image) -> torch.Tensor:
        image = image.resize((512, 512))
        z = self.vqvae_encoder(transforms(image))
        codes = self.quantizer(z)
        return codes.flatten()

    def decode(self, codes: torch.Tensor) -> PIL.Image:
        z_q = self.quantizer.lookup(codes.reshape(32, 32))
        return self.vqvae_decoder(z_q)
```

### 3.2 训练稳定化技术

混合模态早期融合面临的核心挑战：图像 Token 与文本 Token 的**梯度尺度严重失配**，导致训练不稳定。Chameleon 的解决方案：

**① 查询-键归一化（QK-Norm）**：
```python
class StableMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.q_norm = nn.RMSNorm(d_model // num_heads)
        self.k_norm = nn.RMSNorm(d_model // num_heads)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x):
        B, T, D = x.shape
        H = self.attn.num_heads
        q = x.reshape(B, T, H, D//H)
        k = x.reshape(B, T, H, D//H)
        q = self.q_norm(q).reshape(B, T, D)
        k = self.k_norm(k).reshape(B, T, D)
        return self.attn(q, k, x)
```

**② 独立的模态 LayerNorm**：
为图像 Token 和文本 Token 使用独立的 LayerNorm 参数，避免分布差异导致归一化失效：
```python
class ModalityAwareLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.text_norm = nn.LayerNorm(d_model)
        self.image_norm = nn.LayerNorm(d_model)

    def forward(self, x, modality_ids):
        out = torch.zeros_like(x)
        text_mask = modality_ids == 0
        image_mask = modality_ids == 1
        if text_mask.any():
            out[text_mask] = self.text_norm(x[text_mask])
        if image_mask.any():
            out[image_mask] = self.image_norm(x[image_mask])
        return out
```

**③ 分阶段训练课程（Curriculum Training）**：
```
阶段 1: 纯文本预训练（稳定文本能力）
阶段 2: 文本+图像理解（图文联合预训练）
阶段 3: 图文混合生成（统一生成微调）
阶段 4: 指令跟随对齐（RLHF/DPO）
```

---

## 4. 模型规模与性能

| 模型 | 参数量 | 训练数据 | ImageNet 分类 | COCO 图像描述 |
|---|---|---|---|---|
| Chameleon-7B | 7B | 2.9T Token | — | SOTA（当时） |
| Chameleon-34B | 34B | 5.6T Token | — | 超越 GPT-4V（部分任务）|

**文本任务**（纯文本评测）：
- Chameleon-34B 在 MMLU、HumanEval 等基准上超越 Llama-2-70B，与 Mixtral 8x7B 持平。

---

## 5. 使用方法

```python
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

model = ChameleonForConditionalGeneration.from_pretrained(
    "facebook/chameleon-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

import PIL.Image
image = PIL.Image.open("photo.jpg")

prompt = "这张图片描述了什么？<image>"
inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 6. 最新进展（2024-2025）

### 6.1 Emu3（智源研究院，2024）
- 继承 Chameleon 早期融合路线，将视频也纳入统一 Token 空间，实现文本+图像+视频的 Any-to-Any 生成，开源 8B 模型。

### 6.2 BLIP-3 / xGen-MM（Salesforce, 2024）
- 使用 Chameleon 风格的早期融合 + 大规模多模态预训练数据，在多模态理解基准上超越大多数商业模型。

### 6.3 LWM（Large World Model，UC Berkeley, 2024）
- 将 Chameleon 架构扩展至长视频理解（1M Token 上下文），证明早期融合在极长多模态序列处理中的可行性。

### 6.4 对 Tri-Transformer 的技术启示
- **QK-Norm**：应在 C-Transformer 和 I/O-Transformer 中广泛采用，防止多模态训练时的梯度爆炸。
- **分阶段课程训练**：Phase 1 → Phase 2 → Phase 3 的渐进式训练路线图与 Chameleon 的课程训练高度一致。
- **独立模态归一化**：在多模态融合层使用独立 LN 参数，可显著提升训练稳定性。
