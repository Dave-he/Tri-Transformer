# VQ-GAN（向量量化生成对抗网络）

## 1. 概述

VQ-GAN（Vector Quantized Generative Adversarial Network），由 Esser et al. 在《Taming Transformers for High-Resolution Image Synthesis》（CVPR 2021，arXiv:2012.09841）中提出。它将**CNN 编码器的局部归纳偏置**与**Transformer 的全局序列建模能力**有机结合，通过向量量化将连续图像/视频特征空间映射为离散视觉 Token，使 Transformer 可以自回归地生成高分辨率视觉内容。

**在 Tri-Transformer 中的角色**：将视频帧或图像编码为离散视觉 Token 序列，供 I-Transformer（视觉输入理解）和 O-Transformer（视觉内容生成）处理，是多模态 Any-to-Any Token 化的视觉分支。

---

## 2. 架构详解

### 2.1 整体流程

```
原始图像 x ∈ R^{H×W×3}
        ↓
   [CNN Encoder E]
   残差块 + 降采样（降低分辨率 f 倍）
        ↓
   连续特征图 ẑ ∈ R^{H/f × W/f × nz}
        ↓
   [向量量化层 VQ]
   查找最近邻码本向量
        ↓
   量化特征 z_q ∈ R^{H/f × W/f × nz}
   离散码  idx ∈ Z^{H/f × W/f}（展平为 Token 序列）
        ↓
   [CNN Decoder G]
   逐步上采样还原
        ↓
   重建图像 x̂

后续: 离散 Token 序列 → [Transformer] → 自回归生成新 idx → Decoder → 图像
```

### 2.2 向量量化（VQ）层

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0/num_embeddings, 1.0/num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        z: [B, H, W, D] 连续特征（需先将通道维移到最后）
        """
        B, H, W, D = z.shape
        z_flat = z.reshape(-1, D)

        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding.weight.T
                + self.embedding.weight.pow(2).sum(1))

        encoding_indices = dist.argmin(dim=1)
        z_q = self.embedding(encoding_indices).reshape(B, H, W, D)

        loss = self.beta * F.mse_loss(z_q.detach(), z) \
             + F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()
        return z_q, encoding_indices.reshape(B, H, W), loss
```

### 2.3 训练损失组合

```
L_VQ-GAN = L_recon        （感知损失：VGG 特征匹配）
          + λ_adv · L_adv  （图像块判别器 GAN 损失）
          + λ_vq  · L_vq   （VQ 承诺损失）
```

**判别器自适应权重**：VQ-GAN 的关键创新，动态计算 GAN 权重 $\lambda = \nabla_{G_L}[L_{recon}] / (\nabla_{G_L}[L_{adv}] + \epsilon)$，平衡重建质量与生成清晰度。

### 2.4 压缩率选择

| 降采样率 f | Token 数（256×256图像） | 信息密度 | 适用场景 |
|---|---|---|---|
| 4 | 4096 (64×64) | 高 | 精细生成 |
| 8 | 1024 (32×32) | 中 | 标准生成（VQ-GAN 默认） |
| 16 | 256 (16×16) | 低 | 语义级理解 |
| 32 | 64 (8×8) | 极低 | 全局构图规划 |

---

## 3. 视频 Token 化扩展

VQ-GAN 原始设计针对静态图像，视频场景需要时序扩展：

### 3.1 逐帧 VQ-GAN

最简单的方法，对每帧独立量化：

```python
class VideoVQGAN:
    def __init__(self, vqgan: VQGANModel):
        self.vqgan = vqgan

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, T, C, H, W]
        返回: tokens [B, T, H/f, W/f]
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        _, tokens, _ = self.vqgan.encode(frames)
        return tokens.reshape(B, T, *tokens.shape[1:])
```

### 3.2 3D VQ-GAN / VideoGPT

引入时序卷积，对时空体素（Spatiotemporal Volume）进行 3D 量化：

```
视频 [B, T, C, H, W]
   ↓ 3D CNN Encoder（时间降采样 4×，空间降采样 8×）
时空特征 [B, T/4, H/8, W/8, D]
   ↓ VQ 量化
时空 Token 序列（用于自回归生成）
```

---

## 4. 使用方法

### 4.1 使用预训练 VQ-GAN（taming-transformers）

```python
git clone https://github.com/CompVis/taming-transformers
pip install -e .

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

config = OmegaConf.load("configs/imagenet_vqgan_f8_n256.yaml")
model = VQModel(**config.model.params)

state_dict = torch.load("vqgan_imagenet_f8_16384.pt")
model.load_state_dict(state_dict["state_dict"])
model.eval()

img = torch.randn(1, 3, 256, 256)
quant, loss, info = model.encode(img)
tokens = info[2].reshape(1, -1)
print(f"Visual tokens: {tokens.shape}")

reconstructed = model.decode(quant)
```

### 4.2 使用 HuggingFace 托管的 VQ-GAN

```python
from transformers import VQModel

vqvae = VQModel.from_pretrained("openai/dall-e")
```

---

## 5. 最新进展（2024-2025）

### 5.1 VQVAE-2 → FSQ（有限标量量化）
- **FSQ（Finite Scalar Quantization，Google 2023）**：用有界标量代替码本查找，无需直通梯度估计，训练更稳定，码本利用率更高（近 100%）。

### 5.2 Magvit-2（Google 2023/2024）
- 基于 3D VQ 的视频生成 Tokenizer，使用 Look-Up Free Quantization（LFQ）。
- 在视频生成任务上显著优于先前方案，被 Open-Sora、COSMOS 等视频大模型采用。

### 5.3 COSMOS Tokenizer（NVIDIA, 2024）
- NVIDIA 发布的开源多模态视觉 Tokenizer，支持图像和视频，提供连续（VAE）和离散（FSQ）两种变体，是 Tri-Transformer 视觉 Token 化的首选开源工具。

### 5.4 LlamaGen（2024）
- 将高质量 VQ-GAN Token 化与 LLaMA 架构结合，证明自回归语言模型可以达到扩散模型级别的图像生成质量，验证了视觉 Token + LLM = 可行的视觉生成范式。

### 5.5 与 Tri-Transformer 集成建议
- Phase 3 全模态阶段：推荐使用 COSMOS Tokenizer（开源、高质量、支持视频）。
- 离散视觉 Token 与音频 Token、文本 Token 共享 `d_model` 嵌入空间，通过模态标识符（Modality Embedding）区分。
