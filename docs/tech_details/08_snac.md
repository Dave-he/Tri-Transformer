# SNAC（多尺度神经音频编解码器）

## 0. 结论先行

- **核心创新**：多尺度残差量化——不同量化层工作在不同时间分辨率，低层覆盖粗粒度语音结构（音素级），高层覆盖细粒度声学细节（基频、音色），实现语义与声学信息的自然分离。
- **与 EnCodec 的关键差异**：同等码率下 SNAC 保留更丰富的语音细节（尤其低码率场景）；粗粒度 Token（12.5 Token/s）天然适合接入语义建模层，细粒度 Token 适合逐步细化生成。
- **工程推荐**：Phase 2 音频对话阶段优先评估 SNAC 24kHz 版本；粗粒度 Token 接入 I-Transformer 语义编码层，细粒度 Token 由 O-Transformer Streaming Decoder 逐步生成（"先规划后细化"）。
- **Tri-Transformer 中的角色**：EnCodec 的替代音频 Token 化方案，在低码率高语音质量的对话场景中具备竞争优势；层次化 Token 结构与 Tri-Transformer 的"粗→细"生成架构高度契合。

---

## 1. 概述

SNAC（Multi-Scale Neural Audio Codec）是一种层次化神经音频编解码器，核心创新在于**多尺度残差量化**：不同的量化层工作在不同的时间分辨率上，低层码本覆盖粗粒度语音结构（音素级），高层码本覆盖细粒度声学细节（基频、音色），实现语义与声学信息的自然分离。

SNAC 被 Orpheus TTS、Hertz-Dev 等开源实时语音模型所采用，相比 EnCodec 在低码率下保留更丰富的语音细节，且天然支持流式解码。

**在 Tri-Transformer 中的角色**：作为 EnCodec 的替代音频 Token 化方案，在对话场景（低码率高语音质量）中有竞争优势。

---

## 2. 架构详解

### 2.1 多尺度量化结构

SNAC 与 EnCodec 的关键区别在于时间分辨率：

```
EnCodec（均匀时间分辨率）：
T_audio = 24000 samples (1s, 24kHz)
↓ 降采样 320×
T_frames = 75 帧/秒
量化器 1: 75 Token/s  ─┐
量化器 2: 75 Token/s  ─┤ 所有量化器相同时间分辨率
...                     ─┘

SNAC（多尺度时间分辨率）：
量化器 1 (粗粒度): 12.5 Token/s  ← 捕获音节/词级结构
量化器 2 (中粒度): 25  Token/s   ← 捕获音素级细节
量化器 3 (细粒度): 50  Token/s   ← 捕获声调、音色
量化器 4 (超细):   100 Token/s   ← 捕获短时声学特征
```

### 2.2 层次化 Token 树

多尺度结构天然形成 Token 树，适合自回归生成（先生成粗粒度 Token，再细化）：

```
t=0                 t=1
粗粒度:  [  A  ]         [  B  ]
中粒度:  [ a1 ][ a2 ]   [ b1 ][ b2 ]
细粒度:  [a11][a12][a21][a22] [b11][b12][b21][b22]
```

### 2.3 与 EnCodec 对比

| 特性 | EnCodec | SNAC |
|---|---|---|
| 量化时间分辨率 | 所有层相同（75 Token/s） | 层间不同（12.5~100 Token/s） |
| Token 数/秒 | N × 75（N=量化器数） | 层求和（约150~300 Token/s） |
| 语义信息分层 | 无 | 粗层含更多语义 |
| 流式解码 | 支持 | 支持（更优） |
| 低码率音质 | 良好 | 更好 |
| 代表使用 | VALL-E, AudioGen | Orpheus TTS, Hertz-Dev |

---

## 3. 流式解码机制

SNAC 的多尺度结构允许增量解码：粗粒度 Token 到达即可开始解码框架，细粒度 Token 补充细节，实现比 EnCodec 更低的首包延迟：

```python
class SNACStreamingDecoder:
    """
    SNAC 流式解码器：粗粒度 Token 先驱动解码，细粒度 Token 逐步细化
    """
    def __init__(self, model):
        self.model = model
        self.coarse_buffer = []
        self.fine_buffers = [[] for _ in range(3)]

    def push_coarse(self, coarse_token: int):
        """粗粒度 Token 到达，触发初步解码"""
        self.coarse_buffer.append(coarse_token)
        if len(self.coarse_buffer) >= 2:
            return self._decode_chunk()
        return None

    def push_fine(self, fine_tokens: list, level: int):
        """细粒度 Token 到达，精化已解码音频"""
        self.fine_buffers[level].extend(fine_tokens)

    def _decode_chunk(self):
        coarse = torch.tensor(self.coarse_buffer[-2:])
        fine = [torch.tensor(buf[-4:]) for buf in self.fine_buffers]
        return self.model.decode_hierarchical(coarse, fine)
```

---

## 4. 使用方法

### 4.1 安装

```bash
pip install snac
```

### 4.2 编解码

```python
import torch
import torchaudio
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

audio, sr = torchaudio.load("speech.wav")
audio = torchaudio.functional.resample(audio, sr, 24000)
audio = audio.unsqueeze(0).cuda()

with torch.inference_mode():
    codes = model.encode(audio)
    audio_reconstructed = model.decode(codes)

print(f"Codes: {[c.shape for c in codes]}")
```

### 4.3 与语言模型集成（Orpheus 风格）

```python
SNAC_VOCAB_OFFSET = [0, 4096, 8192, 12288]

def snac_codes_to_tokens(codes: list[torch.Tensor]) -> list[int]:
    """将多尺度 SNAC 码交织为单一 Token 序列（适配 LLM vocab）"""
    tokens = []
    max_t = max(c.size(1) for c in codes)
    for t in range(max_t):
        for level, code in enumerate(codes):
            step = max_t // code.size(1)
            if t % step == 0:
                idx = t // step
                if idx < code.size(1):
                    tokens.append(code[0, idx].item() + SNAC_VOCAB_OFFSET[level])
    return tokens
```

---

## 5. 最新进展（2024-2025）

### 5.1 Orpheus TTS（2025，Canopy Labs）
- 基于 SNAC 24kHz 编解码器 + Llama-3B 骨干的开源实时 TTS。
- 推理速度达实时 3×，支持流式输出，延迟 < 200ms。

### 5.2 Hertz-Dev（2024）
- 首个完全开源的实时语音对话模型（类 GPT-4o），采用 SNAC 作为音频 Token 化方案，实现语音全双工交互。

### 5.3 WavTokenizer（2024）
- 将 SNAC 多尺度思想进一步发展：单层量化器但超大码本（40960），0.9 kbps 极低码率下仍保持高音质，减少 LLM 建模的序列长度。

### 5.4 与 Tri-Transformer 的集成建议
- Phase 2 音频对话阶段优先评估 SNAC，尤其是 24kHz 版本。
- 粗粒度 Token（12.5 Token/s）适合接入 I-Transformer 的语义编码层。
- 细粒度 Token 可由 O-Transformer Streaming Decoder 逐步生成，符合"先规划后细化"的反向 Enc-Dec 架构。

---

## 6. 与 Tri-Transformer 的关联

| SNAC 组件 | Tri-Transformer 对应 | 说明 |
|---|---|---|
| 粗粒度量化层（12.5 Token/s） | **I-Transformer** 语义编码层 | 捕获音素级语音结构，供 C-Transformer 语义理解 |
| 细粒度量化层（高分辨率） | **O-Transformer** Streaming Decoder | 逐层细化声学细节，实现流式高质量语音输出 |
| 多尺度 RVQ 结构 | C-Transformer 层次化控制 | 与"先规划后细化"的 Enc-Dec 架构天然契合 |
| 流式解码能力 | 实时音频输出 | 延迟 < 200ms，支持 Phase 2 实时对话场景 |
