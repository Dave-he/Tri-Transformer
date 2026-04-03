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
- I-Transformer 与 O-Transformer 的"可插拔大模型"可直接复用 AnyGPT 或 Emu3 的多模态 Tokenizer 生态。
