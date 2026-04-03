# AnyGPT & Any-to-Any 多模态范式

## 0. 结论先行

- **核心范式**：通过"数据层统一"（而非架构改造），将多种模态的离散 Token 统一在标准 LLM 框架中，实现任意模态输入→任意模态输出，无需针对不同任务组合多个专用模型。
- **关键瓶颈**：模态 Token 速率失配（文本 ~10 Token/s vs 音频 ~75 Token/s vs 视频 ~25 Token/s × fps）导致序列长度急剧增长，需配合分层 Token 化（粗粒度语义 Token + 细粒度声学 Token）缓解。
- **工程推荐**：模态边界标记（`<image>...</image>`，`<audio>...</audio>`）+ 模态嵌入（Modality Embedding）+ 统一嵌入矩阵，参考 AnyGPT/Emu3 实现；优先使用已有开源 Tokenizer（EnCodec/SNAC for audio, COSMOS for video）而非从头训练。
- **Tri-Transformer 中的角色**：为 I/O 端共享离散 Token 空间的设计（BPE + Codec + VQ）提供验证依据和词表设计参考；I/O 端"可插拔大模型"可直接复用 AnyGPT/Emu3 的多模态 Tokenizer 生态。

---

## 1. 概述

**Any-to-Any 范式**是指单一模型能够接受任意模态（文本、语音、图像、视频、音乐）的输入，并生成任意模态的输出，无需针对不同任务组合多个专用模型。

**AnyGPT**（浙江大学 & 字节跳动，2024，arXiv:2402.12226）是首批全面验证这一范式的开源工作之一，通过**数据层统一**（而非架构改造），将多种模态的离散 Token 统一在标准 LLM 框架中进行任意模态到任意模态的生成。

**在 Tri-Transformer 中的角色**：为 I/O 端共享离散 Token 空间的设计（BPE 文本 Token + Codec 音频 Token + VQ 视觉 Token 共用 `d_model` 嵌入空间）提供验证依据和词表设计参考。

---

## 2. AnyGPT 架构详解

### 2.1 核心设计哲学

AnyGPT 的关键洞察：**任意模态的信息，只要能被离散化为 Token 序列，就可以被现有 LLM 架构无缝处理**，无需任何架构改变。

```
多模态输入统一流程：
文本 "你好"    ──→ BPE Token  [1234, 5678]
语音 (audio)  ──→ Codec Token [A001, A002, A003, ...]
图像 (image)  ──→ VQ Token   [V001, V002, ..., V256]
音乐 (music)  ──→ EnCodec    [M001, M002, ...]
                                 ↓
                    统一嵌入 Embedding
                                 ↓
                    标准 LLM Transformer
                                 ↓
                    预测下一个 Token（任意模态）
```

### 2.2 模态处理组件

| 模态 | Tokenizer | 词表大小 | 时间分辨率 |
|---|---|---|---|
| 文本 | LLaMA2 BPE | 32,000 | — |
| 语音 | SpeechTokenizer（语义层） | 1,000 | 50 Token/s |
| 图像 | SEED Tokenizer（离散视觉 Token） | 8,192 | 32×32 = 1024 Token/图 |
| 音乐 | EnCodec | 2,048 | 75 Token/s |

### 2.3 统一词表设计

```python
class AnyGPTVocabulary:
    TEXT_VOCAB_SIZE = 32000
    SPEECH_VOCAB_SIZE = 1000
    IMAGE_VOCAB_SIZE = 8192
    MUSIC_VOCAB_SIZE = 2048

    SPEECH_OFFSET = TEXT_VOCAB_SIZE
    IMAGE_OFFSET = SPEECH_OFFSET + SPEECH_VOCAB_SIZE
    MUSIC_OFFSET = IMAGE_OFFSET + IMAGE_VOCAB_SIZE
    TOTAL_VOCAB_SIZE = MUSIC_OFFSET + MUSIC_VOCAB_SIZE

    SPECIAL_TOKENS = {
        "<|speech_start|>": TOTAL_VOCAB_SIZE,
        "<|speech_end|>": TOTAL_VOCAB_SIZE + 1,
        "<|image_start|>": TOTAL_VOCAB_SIZE + 2,
        "<|image_end|>": TOTAL_VOCAB_SIZE + 3,
        "<|music_start|>": TOTAL_VOCAB_SIZE + 4,
        "<|music_end|>": TOTAL_VOCAB_SIZE + 5,
    }

def encode_speech(audio_codes: torch.Tensor) -> list:
    return [c + AnyGPTVocabulary.SPEECH_OFFSET for c in audio_codes.tolist()]

def encode_image(vq_codes: torch.Tensor) -> list:
    return [c + AnyGPTVocabulary.IMAGE_OFFSET for c in vq_codes.tolist()]
```

### 2.4 多模态指令数据集

AnyGPT 的重要贡献：构建了首个大规模 Any-to-Any 多模态对话数据集（108K 样本），包含：

```
对话类型示例：
- 文本 → 语音（TTS 对话）
- 语音 → 文本（ASR 对话）
- 图像 + 文本 → 文本（图文问答）
- 文本 → 图像（文生图）
- 语音 → 语音（语音对话）
- 图像 → 语音（看图说话）
- 文本 → 音乐（文生音乐）
- 多轮混合模态对话
```

---

## 3. 多模态 Token 对齐挑战

### 3.1 模态速率失配问题

不同模态的 Token 生成速率差异巨大：

| 模态 | Token 速率 | 10秒的 Token 数 |
|---|---|---|
| 文本（口语速率） | ~10 Token/s | ~100 |
| 语音（Codec） | 75 Token/s | 750 |
| 视频（VQ，10fps） | 1024 Token/帧 × 10 fps | 102,400 |

解决方案：
1. **降采样**：减少视频帧率（1fps）或使用更粗粒度的 Token。
2. **层次建模**：Moshi 风格的双流建模，语义 Token 和声学 Token 分层生成。
3. **压缩编码**：SNAC 多尺度量化减少声学 Token 总数。

### 3.2 模态边界标记

```python
def build_multimodal_sequence(items: list[dict]) -> list[int]:
    """
    items: [{"type": "text", "content": "..."}, 
            {"type": "speech", "tokens": [...]},
            {"type": "image", "tokens": [...]}]
    """
    vocab = AnyGPTVocabulary
    tokens = []
    for item in items:
        if item["type"] == "text":
            tokens.extend(bpe_encode(item["content"]))
        elif item["type"] == "speech":
            tokens.append(vocab.SPECIAL_TOKENS["<|speech_start|>"])
            tokens.extend(encode_speech(item["tokens"]))
            tokens.append(vocab.SPECIAL_TOKENS["<|speech_end|>"])
        elif item["type"] == "image":
            tokens.append(vocab.SPECIAL_TOKENS["<|image_start|>"])
            tokens.extend(encode_image(item["tokens"]))
            tokens.append(vocab.SPECIAL_TOKENS["<|image_end|>"])
    return tokens
```

---

## 4. 使用方法

### 4.1 AnyGPT 官方代码

```bash
git clone https://github.com/OpenMOSS/AnyGPT
pip install -r requirements.txt

python inference.py \
    --model_path AnyGPT-base \
    --input_type speech \
    --output_type text \
    --input audio/question.wav
```

### 4.2 自定义 Any-to-Any 模型骨架

```python
from transformers import LlamaForCausalLM, LlamaConfig

class AnyToAnyModel(nn.Module):
    def __init__(self, base_model_name: str, extra_vocab_size: int):
        super().__init__()
        self.backbone = LlamaForCausalLM.from_pretrained(base_model_name)
        orig_vocab = self.backbone.config.vocab_size

        self.backbone.resize_token_embeddings(orig_vocab + extra_vocab_size)

        new_emb_weight = self.backbone.model.embed_tokens.weight.data
        nn.init.normal_(new_emb_weight[orig_vocab:], mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.backbone(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
```

---

## 5. 最新进展（2024-2025）

### 5.1 Unified-IO 2（Allen AI，2024）
- 将文本、图像、音频、视频、动作等 10 余种模态统一为离散 Token，是目前最全面的 Any-to-Any 开源模型。

### 5.2 Emu3（智源研究院，2024）
- 全离散 Token（文本 + 视觉 VQ）的自回归多模态模型，去除所有传统视觉编码器，直接在统一 Token 空间上训练。

### 5.3 Show-o（2024）
- 将自回归（文本理解）与扩散（图像生成）在同一模型中统一，文本 Token 自回归生成，图像 Token 用掩码扩散生成，是 Any-to-Any 范式的混合路径探索。

### 5.4 与 Tri-Transformer 的关联
- Tri-Transformer 的 Token 空间设计与 AnyGPT 高度一致：BPE 文本 + EnCodec/SNAC 音频 + VQ-GAN/SigLIP 视觉，通过模态标识符 Embedding（Modality Embedding）区分。
- I-Transformer 与 O-Transformer 的\"可插拔大模型\"可直接复用 AnyGPT 或 Emu3 的多模态 Tokenizer 生态。

---

## 6. 深度扩展（2024-2025 工程实践）

### 6.1 Emu3 完整架构与推理实践（智源, 2024）

Emu3（arXiv:2409.18869）彻底去除独立视觉编码器，直接在统一离散 Token 空间上自回归训练，实现了真正意义上的"Next Token Prediction 统一多模态"：

```python
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/Emu3-Gen", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "BAAI/Emu3-Gen",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

GENERATION_PROMPT = (
    "<|image start|><|image placeholder|><|image end|>"
    "\n请详细描述这张图片："
)
inputs = tokenizer(
    GENERATION_PROMPT,
    return_tensors="pt",
    add_special_tokens=True,
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=1.0,
        suppress_tokens=tokenizer.encode("<|image start|>"),
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Emu3 统一 Token 空间设计**：

```
Emu3 词表设计：
  文本 Token:    0 ~ 32000    (LLaMA-3 BPE)
  视觉 Token:    32001 ~ 64000 (VQ-VAE codebook, 32K条目)
  特殊 Token:    64001+        (<|image start|>, <|image end|>, ...)
  总词表大小:    ~64K
```

```python
class Emu3TokenSpace:
    """Emu3 统一 Token 空间管理"""
    TEXT_VOCAB_SIZE = 32000
    VISUAL_VOCAB_SIZE = 32000
    TOTAL_VOCAB_SIZE = TEXT_VOCAB_SIZE + VISUAL_VOCAB_SIZE + 100

    IMAGE_START = TEXT_VOCAB_SIZE + VISUAL_VOCAB_SIZE
    IMAGE_END = IMAGE_START + 1
    IMAGE_NEWLINE = IMAGE_START + 2

    @staticmethod
    def visual_token_to_id(code: int) -> int:
        return Emu3TokenSpace.TEXT_VOCAB_SIZE + code

    @staticmethod
    def id_to_visual_token(token_id: int) -> int:
        return token_id - Emu3TokenSpace.TEXT_VOCAB_SIZE
```

### 6.2 Janus-Pro：解耦理解与生成的 Any-to-Any（DeepSeek, 2025）

Janus-Pro（arXiv:2501.17811）的核心创新是**解耦视觉理解和视觉生成的编码路径**，解决了统一多模态模型在理解精度和生成质量上的固有权衡：

```
传统 Any-to-Any（AnyGPT / Emu3）：
  图像输入 → [同一 VQ 编码器] → 离散视觉 Token → LLM → 理解/生成

Janus-Pro 解耦方案：
  图像输入（理解任务）→ [SigLIP 连续编码器] → 连续特征 → LLM → 文本输出
  图像输出（生成任务）→ [VQ 离散解码器] → 离散 Token 序列 → LLM → VQ 解码 → 图像
```

```python
class JanusPro(torch.nn.Module):
    """Janus-Pro 解耦多模态架构骨架"""
    def __init__(self, llm_backbone, vision_encoder, vq_decoder):
        super().__init__()
        self.llm = llm_backbone
        self.vision_enc = vision_encoder
        self.vq_dec = vq_decoder
        self.understanding_proj = torch.nn.Linear(
            vision_encoder.config.hidden_size,
            llm_backbone.config.hidden_size
        )
        self.generation_head = torch.nn.Linear(
            llm_backbone.config.hidden_size,
            vq_decoder.codebook_size
        )

    def forward_understanding(self, pixel_values, text_ids):
        vis_features = self.vision_enc(pixel_values).last_hidden_state
        vis_tokens = self.understanding_proj(vis_features)
        text_emb = self.llm.model.embed_tokens(text_ids)
        combined = torch.cat([vis_tokens, text_emb], dim=1)
        return self.llm(inputs_embeds=combined)

    def forward_generation(self, text_ids, visual_token_ids=None):
        if visual_token_ids is not None:
            gen_emb = self.llm.model.embed_tokens(
                visual_token_ids + self.llm.config.vocab_size
            )
            text_emb = self.llm.model.embed_tokens(text_ids)
            inputs_embeds = torch.cat([text_emb, gen_emb], dim=1)
        else:
            inputs_embeds = self.llm.model.embed_tokens(text_ids)
        logits = self.llm(inputs_embeds=inputs_embeds).logits
        visual_logits = self.generation_head(logits)
        return visual_logits
```

**Janus-Pro 在 GenAI 基准上的性能**：

| 模型 | GenEval ↑ | DPG-Bench ↑ | MMStar ↑ | 参数量 |
|------|-----------|-------------|---------|--------|
| Janus-1.3B | 0.769 | 71.2 | 55.3 | 1.3B |
| Janus-Pro-7B | 0.848 | 80.0 | 59.8 | 7B |
| DALL-E 3 | 0.813 | 83.5 | — | — |
| SD 3.5 | 0.768 | 83.0 | — | — |

### 6.3 Show-o：自回归与扩散的混合统一（2024）

Show-o（arXiv:2408.12528）在同一 Transformer 中实现自回归（文本）+ 掩码扩散（图像）的双模态生成：

```python
class ShowoUnifiedTransformer(torch.nn.Module):
    """Show-o 混合生成模式 Transformer"""
    def __init__(self, d_model=2048, vocab_size=58498):
        super().__init__()
        from transformers import PhiForCausalLM, PhiConfig
        config = PhiConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=24,
            num_attention_heads=32,
        )
        self.transformer = PhiForCausalLM(config)
        self.MASK_TOKEN_ID = vocab_size - 1

    def forward_text(self, input_ids):
        return self.transformer(input_ids=input_ids)

    def forward_image_masked_diffusion(
        self, noisy_tokens, noise_level: float
    ):
        mask_ratio = noise_level
        mask = torch.rand_like(noisy_tokens.float()) < mask_ratio
        noisy_tokens = noisy_tokens.masked_fill(mask, self.MASK_TOKEN_ID)
        logits = self.transformer(input_ids=noisy_tokens).logits
        return logits, mask

    def denoise_step(self, tokens, t: int, T: int = 1000):
        noise_level = 1.0 - t / T
        logits, mask = self.forward_image_masked_diffusion(tokens, noise_level)
        predicted = logits.argmax(-1)
        tokens = torch.where(mask, predicted, tokens)
        return tokens
```

### 6.4 Tri-Transformer Any-to-Any Token 统一空间设计

基于以上三种架构的学习，Tri-Transformer 的最优 Any-to-Any Token 空间设计方案：

```python
class TriTransformerUnifiedTokenSpace:
    """
    Tri-Transformer 统一 Token 空间
    借鉴 AnyGPT 的数据层统一 + Janus-Pro 的解耦编码思想
    """
    TEXT_VOCAB_SIZE = 152064
    AUDIO_VOCAB_OFFSET = 152064
    AUDIO_VOCAB_SIZE = 4096
    VISUAL_VOCAB_OFFSET = 156160
    VISUAL_VOCAB_SIZE = 32768
    CONTROL_VOCAB_OFFSET = 188928
    CONTROL_VOCAB_SIZE = 256
    TOTAL_VOCAB_SIZE = 189184

    SPECIAL_TOKENS = {
        "<|text_start|>": CONTROL_VOCAB_OFFSET + 0,
        "<|text_end|>": CONTROL_VOCAB_OFFSET + 1,
        "<|audio_start|>": CONTROL_VOCAB_OFFSET + 2,
        "<|audio_end|>": CONTROL_VOCAB_OFFSET + 3,
        "<|visual_start|>": CONTROL_VOCAB_OFFSET + 4,
        "<|visual_end|>": CONTROL_VOCAB_OFFSET + 5,
        "<|interrupt|>": CONTROL_VOCAB_OFFSET + 6,
    }

    @classmethod
    def text_to_id(cls, token_id: int) -> int:
        return token_id

    @classmethod
    def audio_to_id(cls, codec_id: int) -> int:
        return cls.AUDIO_VOCAB_OFFSET + codec_id

    @classmethod
    def visual_to_id(cls, vq_id: int) -> int:
        return cls.VISUAL_VOCAB_OFFSET + vq_id
