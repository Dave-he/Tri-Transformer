# EnCodec（神经音频编解码器）

## 0. 结论先行

- **核心作用**：端到端训练的流式卷积网络 + 残差向量量化（RVQ），将连续音频波形压缩为层次化离散声学 Token 序列，是基于 Codec 的语音语言模型（VALL-E、Moshi 等）的标准音频离散化基础设施。
- **关键参数**：24kHz 单声道，8 个量化器（N=8），每个码本 1024 个条目，Token 速率 75 Token/s，总码率约 6 kbps；推理速度优于实时（CPU 可跑）。
- **工程推荐**：直接使用 `encodec` PyPI 包的预训练模型；音频 Token 与文本 BPE Token 共享嵌入空间时，建议追加独立嵌入层而非共享嵌入矩阵（避免梯度干扰）。
- **Tri-Transformer 中的角色**：I-Transformer 音频流输入的预处理（波形 → Codec Token），O-Transformer 输出的后处理（Codec Token → 波形）；是 Phase 2 音频对话阶段的默认方案。

---

## 1. 概述

EnCodec 是 Meta AI 于 2022 年发布的高保真神经音频编解码器（arXiv:2210.13438），通过端到端训练的流式卷积 Encoder-Decoder 网络 + 残差向量量化（Residual Vector Quantization, RVQ），将连续音频波形压缩为层次化**离散声学 Token** 序列，同时保持实时性（编解码速度优于实时）。

EnCodec 是当前几乎所有基于 Codec 的语音/音频语言模型（VALL-E、Moshi、SoundStorm、AudioGen 等）的标准音频离散化基础设施。

**在 Tri-Transformer 中的角色**：I-Transformer 接收音频流前的预处理（音频 → Codec Token），O-Transformer 输出后的后处理（Codec Token → 音频波形）。

---

## 2. 架构详解

### 2.1 整体流程

```
输入音频波形 x ∈ R^{T}
        ↓
   [EnCodec Encoder]
   Conv1D + EncoderBlock×4（逐步降采样 2×2×8×8=256倍）
        ↓
   连续潜在表征 z ∈ R^{T/320 × D}
        ↓
   [残差向量量化 RVQ]
   N 个量化器，每个码本大小 K=1024
        ↓
   离散码序列 codes ∈ Z^{T/320 × N}
        ↓
   [EnCodec Decoder]
   Conv1D + DecoderBlock×4（逐步升采样）
        ↓
   重建音频波形 x̂
```

### 2.2 残差向量量化（RVQ）

RVQ 是 EnCodec 的核心，通过多层量化层次化捕获音频信息：

```python
class ResidualVectorQuantizer(nn.Module):
    """
    N 层残差向量量化器
    第 i 层量化第 i-1 层的残差，逐层精化
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=128):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, dim) for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor):
        """z: [B, T, D] 连续潜在表征"""
        residual = z
        all_codes = []
        all_quantized = []

        for vq in self.quantizers:
            quantized, codes = vq(residual)
            residual = residual - quantized
            all_codes.append(codes)
            all_quantized.append(quantized)

        z_q = sum(all_quantized)
        codes = torch.stack(all_codes, dim=-1)
        return z_q, codes


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1/codebook_size, 1/codebook_size)

    def forward(self, z: torch.Tensor):
        """z: [B, T, D] → quantized: [B, T, D], codes: [B, T]"""
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)

        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(1))

        codes = dist.argmin(dim=-1).reshape(B, T)
        quantized = self.codebook(codes)
        return quantized, codes
```

### 2.3 量化器数量与码率的关系

| 量化器数量 N | 码率 (kbps, 24kHz) | 适用场景 |
|---|---|---|
| 2 | 1.5 | 极低码率语音（可懂但音质差） |
| 4 | 3.0 | 低码率语音通话 |
| 8 | 6.0 | 高质量语音（推荐，VALL-E 默认） |
| 12 | 9.0 | 高质量音乐 |
| 16 | 12.0 | 专业音乐制作 |
| 32 | 24.0 | 48kHz 立体声高保真 |

### 2.4 流式架构

EnCodec 的 Encoder 和 Decoder 均使用流式卷积（Causal Conv1D），支持逐帧实时处理：

```python
class CausalConv1d(nn.Module):
    """因果卷积：当前帧只依赖历史帧"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)
```

---

## 3. 训练损失

EnCodec 使用多任务损失组合：

```
L_total = λ_t · L_time    （时域重建损失，L1 + L2）
        + λ_f · L_freq    （多尺度频域损失，对数梅尔频谱 L1 + L2）
        + λ_g · L_adv     （多尺度频谱判别器对抗损失）
        + λ_feat · L_feat  （判别器特征匹配损失）
        + λ_vq · L_vq     （向量量化承诺损失）
```

**损失平衡器（Loss Balancer）**：EnCodec 的创新贡献之一，动态调整各损失权重，使每个损失对梯度的贡献比例保持固定，避免某一损失主导训练。

---

## 4. 使用方法

### 4.1 安装与基本使用

```bash
pip install encodec
```

```python
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.eval()

wav, sr = torchaudio.load("audio.wav")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

with torch.no_grad():
    encoded_frames = model.encode(wav)

codes = torch.cat([e[0] for e in encoded_frames], dim=-1)
print(f"Codec codes shape: {codes.shape}")

with torch.no_grad():
    audio_values = model.decode(encoded_frames)
```

### 4.2 与 Transformer 语言模型集成

```python
class AudioLanguageModel(nn.Module):
    """将 Codec Token 作为输入/输出的音频语言模型"""
    def __init__(self, vocab_size=1024, n_quantizers=8, d_model=512, n_layers=12):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codec_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(n_quantizers)
        ])
        self.transformer = CausalTransformer(d_model, n_layers)
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_quantizers)
        ])

    def forward(self, codec_codes: torch.Tensor):
        """codec_codes: [B, T, N_Q]"""
        x = sum(emb(codec_codes[..., i])
                for i, emb in enumerate(self.codec_embeddings))
        hidden = self.transformer(x)
        logits = [head(hidden) for head in self.output_heads]
        return logits
```

---

## 5. 最新进展（2024-2025）

### 5.1 DAC（Descript Audio Codec，2023）
- 在 EnCodec 基础上引入改进训练策略，在更低码率下达到更高音质，支持 44.1kHz 音乐。
- 已成为 AudioCraft（Meta）的默认 Codec。

### 5.2 SNAC（Multi-Scale Neural Audio Codec，2024）
- 多尺度量化：不同量化层使用不同时间分辨率，低层捕获粗粒度语音结构，高层捕获细节。
- 在 Orpheus TTS、Hertz-Dev 等开源模型中使用，流式解码性能优于 EnCodec。

### 5.3 Single-Codec（2024）
- 探索使用单一量化器（N=1）配合大码本（K=65536），以简化语言建模头部设计。

### 5.4 SemantiCodec / SpeechTokenizer（2024）
- **SpeechTokenizer**：第一个量化器专门捕获语义信息（对齐 HuBERT），后续量化器捕获声学细节，实现语义-声学分离。
- 使得基于 Codec 的 TTS 模型能分别优化内容生成和说话人风格克隆。
