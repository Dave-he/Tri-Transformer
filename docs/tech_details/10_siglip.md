# SigLIP（Sigmoid 损失语言-图像预训练）

## 0. 结论先行

- **核心创新**：用逐对 Sigmoid 损失替代 CLIP 的 Softmax 对比损失，彻底消除对全局批次归一化的需求，每个图文对独立计算，可使用任意批次大小训练，扩展性显著优于 CLIP。
- **工程推荐**：直接加载 HuggingFace 上的 `google/siglip-so400m-patch14-384` 预训练权重，用作视觉编码器；输出 patch token 序列（`[B, N_patches, D]`）通过线性投影对齐 LLM 的 `d_model`。
- **主流多模态 LLM 的标准配置**：PaliGemma、InternVL 2.5、Qwen2-VL 均采用 SigLIP 视觉编码器，是当前开源多模态系统视觉分支的事实标准。
- **Tri-Transformer 中的角色**：O-Transformer Planning Encoder 的视觉编码骨干，提供对齐了自然语言语义的视觉特征空间，用于视频帧的全局规划与跨模态 RAG 查询向量生成。

---

## 1. 概述

SigLIP（Sigmoid Loss for Language Image Pre-Training）由 Google 研究人员（Zhai et al.）于 2023 年发布（arXiv:2303.15343，ICCV 2023 Oral），是对 CLIP 对比学习框架的重要改进。其核心创新是：**用逐对 Sigmoid 损失替代 Softmax 对比损失**，彻底消除了对全局批次归一化的需求，实现了更好的训练扩展性和更低的计算成本。

SigLIP 的视觉编码器被 PaliGemma、InternVL、Qwen2-VL 等主流多模态大模型广泛采用，是当前最强的开源视觉语言对齐骨干之一。

**在 Tri-Transformer 中的角色**：作为视觉帧编码器的预训练骨干，提供对齐了自然语言语义的视觉特征空间，供 O-Transformer Planning Encoder 进行视觉内容的全局规划。

---

## 2. 核心原理：Sigmoid vs. Softmax

### 2.1 CLIP 的 Softmax 对比损失

CLIP 将 N 对图像-文本对中的正样本拉近、负样本推远：

$$L_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]$$

**问题**：Softmax 分母需要全批次所有对的相似度分数，要求批次内所有样本全局可见，限制了分布式训练的灵活性。

### 2.2 SigLIP 的 Sigmoid 损失

SigLIP 将每对图像-文本视为独立的二分类问题：

$$L_{\text{SigLIP}} = -\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\sigma\left(z_{ij} \cdot (2y_{ij} - 1)\right)$$

其中：
- $z_{ij} = t \cdot \langle v_i, w_j \rangle + b$（$v$ 视觉特征，$w$ 文本特征，$t$ 可学习温度，$b$ 偏置）
- $y_{ij} = 1$ 若 $i=j$（正样本），否则 $y_{ij} = -1$（负样本）
- $\sigma(\cdot)$ 为 Sigmoid 函数

**关键优势**：每对样本独立计算 Sigmoid，无需全局 Softmax 归一化，天然支持任意批次大小和分布式并行。

### 2.3 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    def __init__(self, init_temperature: float = 10.0, init_bias: float = -10.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds: [N, D] 归一化图像特征
        text_embeds:  [N, D] 归一化文本特征
        """
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        t = self.log_temperature.exp()
        logits = torch.matmul(image_embeds, text_embeds.T) * t + self.bias

        N = logits.size(0)
        labels = 2 * torch.eye(N, device=logits.device) - 1

        loss = -F.logsigmoid(labels * logits).sum() / N
        return loss
```

---

## 3. 模型架构

SigLIP 的视觉编码器基于 ViT（Vision Transformer），文本编码器基于标准 Transformer：

### 3.1 模型变体

| 模型 | 视觉编码器 | 图像分辨率 | 参数量 | ImageNet 零样本精度 |
|---|---|---|---|---|
| SigLIP-B/16 | ViT-B/16 | 224×224 | 86M | 76.7% |
| SigLIP-L/16 | ViT-L/16 | 224×224 | 307M | 82.7% |
| SigLIP-So400m/14 | ViT-So400m | 224×224 | 400M | 83.1% |
| SigLIP-So400m/14-384 | ViT-So400m | 384×384 | 400M | **84.5%** |
| SigLIP2（2024） | ViT-So400m | 512×512 | 400M+ | > 85% |

### 3.2 视觉特征提取

```python
from transformers import AutoProcessor, SiglipVisionModel

model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

import PIL.Image
image = PIL.Image.open("frame.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    vision_outputs = model(**inputs)

image_features = vision_outputs.last_hidden_state
print(f"Visual patch tokens: {image_features.shape}")
pooled_output = vision_outputs.pooler_output
print(f"Pooled visual feature: {pooled_output.shape}")
```

---

## 4. 与 CLIP 的全面对比

| 维度 | CLIP | SigLIP |
|---|---|---|
| 损失函数 | Softmax 对比损失 | Sigmoid 二分类损失 |
| 批次依赖 | 强（需大批次） | 弱（任意批次大小） |
| 最优批次大小 | 32K-65K | 32K 足够 |
| 小批次性能 | 差 | 好 |
| 负样本处理 | 批次内所有负样本 | 独立二分类 |
| 偏置项 | 无 | 有（可学习，初始化为负） |
| 零样本精度 | 76.2%（ViT-L） | 84.5%（So400m） |
| 代表应用 | OpenAI DALL-E, CLIP | PaliGemma, Qwen2-VL |

---

## 5. 使用方法

### 5.1 作为 Tri-Transformer 视觉编码器

```python
from transformers import SiglipVisionModel, SiglipVisionConfig
import torch.nn as nn

class VisualEncoder(nn.Module):
    """将 SigLIP ViT 视觉编码器集成到 I-Transformer"""
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.proj = nn.Linear(self.siglip.config.hidden_size, d_model)

    def encode_frame(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, 384, 384]
        返回: visual_tokens [B, num_patches, d_model]
        """
        outputs = self.siglip(pixel_values=pixel_values)
        patch_features = outputs.last_hidden_state
        return self.proj(patch_features)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, T, 3, 384, 384]
        返回: video_tokens [B, T*num_patches, d_model]
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        tokens = self.encode_frame(frames)
        return tokens.reshape(B, T * tokens.size(1), -1)
```

### 5.2 图像-文本相似度计算

```python
from transformers import AutoProcessor, SiglipModel

model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

inputs = processor(
    text=["一只猫", "一条狗", "一辆汽车"],
    images=[img1, img2],
    return_tensors="pt",
    padding="max_length"
)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
```

---

## 6. 最新进展（2024-2025）

### 6.1 SigLIP 2（Google, 2024）
- 引入更高分辨率训练（512×512）、多语言扩展、更大模型（600M+）。
- 在多模态理解基准（MMVP、MMBench）上进一步提升。

### 6.2 SigLIP 在主流多模态 LLM 中的应用
- **PaliGemma（Google, 2024）**：SigLIP-So400m + Gemma，Google 第一个开源 VLM。
- **InternVL 2.5（2024）**：SigLIP 视觉编码器 + InternLM 语言模型，中文多模态 SOTA。
- **Qwen2-VL（阿里, 2024）**：改进的 SigLIP 视觉编码器，支持任意分辨率动态视觉 Token。

### 6.3 NaViT（Native Resolution ViT，Google）
- 扩展 SigLIP 至任意分辨率和纵横比输入，通过 Packing 技术将不同分辨率图像混合训练，是 Qwen2-VL 动态分辨率的技术前身。
