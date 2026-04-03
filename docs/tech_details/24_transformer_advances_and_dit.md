# 2020–至今 Transformer 前沿进展与 DiT 架构

## 0. 结论先行（给工程/研究决策者的要点）

过去 5–6 年 Transformer 的关键进展可以归纳为三条主线：

1. **注意力与推理效率工程化**（FlashAttention / KV Cache / Serving）
2. **长上下文与位置编码可外推**（RoPE / ALiBi / YaRN）
3. **以"块"为单位的通用骨干迁移到生成建模**（DiT 等）

在图像生成方向，DiT 的核心贡献不是"把 U-Net 换成 Transformer"这么简单，而是形成了一套可扩展配方：**latent patch tokenization + timestep/条件的 AdaLN-Zero 调制 + 大规模可扩展的 ViT 式堆叠**，并在 ImageNet class-conditional 上展示了随计算量（Gflops）扩大而 FID 稳定下降的 scaling 行为。

**工程建议**：
- 快速搭建并可持续迭代：以 PyTorch 2.x 的 `scaled_dot_product_attention`（SDPA）为默认注意力后端，自动落到 fused kernel（Flash / memory-efficient / math），并在推理/采样侧结合 `torch.compile` 做图融合与 kernel 级优化。
- 服务端高吞吐 LLM：KV cache 的内存管理（PagedAttention / vLLM）是系统级拐点。

---

## 1. Transformer（2020–至今）关键创新脉络

### 1.1 注意力计算：从"算法近似"到"硬件感知的精确 fused kernel"

#### A. 线性/稀疏注意力（算法路径，适合超长序列或特定结构）

| 方法 | 核心思想 | 复杂度 | 适用场景 |
|---|---|---|---|
| **Reformer**（2020） | LSH 局部敏感哈希将全连接注意力近似 | O(L log L) | 极长序列，实现与训练稳定性成本较高 |
| **Longformer**（2020） | 局部滑窗 + 少量全局 token 稀疏模式 | 近线性 | 长文档 NLP，需改注意力模式 |
| **Performer**（2020） | FAVOR+ 核方法随机特征逼近 softmax | 线性 | 超长序列、可容忍近似 |

#### B. FlashAttention 系列（系统/算子路径：不近似，极致 IO 优化）

**FlashAttention（2022）**：
- 以 IO-aware 的方式在 GPU SRAM/HBM 间组织读写
- 做到"精确注意力但显著更快、更省显存"
- 论文：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

**FlashAttention-2（2023）**：
- 改进并行划分与工作分配
- 对 FlashAttention-1 约有 2× 加速，更接近 GEMM 效率上限
- 更好地利用 Tensor Core 与 warp-level 并行

**FlashAttention-3（2024/2025）**：
- 面向 H100 等新硬件，以异步与低精度（FP8）进一步提高吞吐
- BF16/FP8 下显著 TFLOPs 提升（报告约 740 TFLOPs/s on H100）
- Warp 专化异步流水线，详见 [18_flashattention3.md](./18_flashattention3.md)

> **工程含义**：从 2022 起，"高性能注意力"逐渐从模型论文迁移为框架默认能力。PyTorch 2.x 的 SDPA 会根据硬件与形状自动选择实现（可能落到 fused kernel），是搭建 Transformer/DiT 的最稳妥默认项。

---

### 1.2 位置编码与长上下文：从"编码形式"到"可外推扩窗配方"

| 方法 | 核心思想 | 外推能力 | 主要优势 |
|---|---|---|---|
| **RoPE**（2021） | 旋转方式将相对位置信息融入注意力计算 | 有限（需扩窗方法配合） | 相对位置自然编码，成为大模型主流方案 |
| **ALiBi**（2022） | 注意力 logits 上加距离成比例的线性偏置，无显式位置 embedding | 较好 | train short, test long；训练更快、显存更省 |
| **YaRN**（2023） | 针对 RoPE 模型的上下文扩展，以更少 token/步数实现适配 | 强（扩至 128K+） | RoPE 扩窗的高性价比方案 |

**Qwen3 示例**：`rope_theta=1,000,000` + YaRN → 原生 32K，扩展至 128K（详见 [23_qwen3.md](./23_qwen3.md)）

> **工程含义**：若研究目标含"长上下文/长序列"，建议将"位置编码选择"与"扩窗方法"解耦——RoPE/ALiBi 是底座，YaRN 等是扩窗阶段的策略层。

---

### 1.3 推理效率：KV Cache 规模、带宽与服务端内存管理

在自回归解码中，KV cache 通常是吞吐/并发的核心约束。

| 方法 | 核心思想 | 主要收益 | 参考文档 |
|---|---|---|---|
| **GQA**（2023） | Grouped-Query Attention，在 MHA 与 MQA 间折中，用更少 KV head | 推理带宽/显存显著下降，支持从已有 checkpoint uptraining | — |
| **PagedAttention / vLLM**（2023） | KV cache 当作"分页内存"管理，减少碎片与浪费 | 大幅提升服务端吞吐与共享前缀能力 | [20_vllm_pagedattention.md](./20_vllm_pagedattention.md) |
| **Layer-Condensed KV Cache**（2024） | 只缓存少数层的 KV 以显著省内存 | 报告吞吐数量级提升 | — |

> **工程含义**：训练侧常以 FlashAttention/SDPA 降显存；部署侧常以 GQA + PagedAttention 同时解决"带宽与内存碎片"问题。

---

### 1.4 非注意力替代骨干：线性时间序列建模的探索

| 方法 | 核心思想 | 优势 | 主要门槛 |
|---|---|---|---|
| **RetNet**（2023） | retention 机制，支持并行/递归/分块递归三种范式 | 兼顾训练并行与低成本推理 | 生态与工具链兼容度 |
| **Mamba（Selective SSM）**（2023） | 选择性状态空间模型，线性时间序列建模 | 长序列扩展与吞吐优势 | 实现复杂度，CUDA kernel 依赖 |
| **RWKV**（2023） | 结合 Transformer 可并行训练与 RNN 常数复杂度推理 | 部署效率高 | 表达能力上限不确定 |

> **研究含义**：这些路线对"极长序列、低延迟"很有吸引力，但生态、可解释性与现有工具链兼容度仍是主要门槛；短中期内，主流工业栈仍以优化版 Transformer 为主。

---

## 2. DiT（Diffusion Transformer）架构

### 2.1 DiT 解决了什么

DiT（Peebles & Xie, 2022/2023）在 latent diffusion 框架内，用 Transformer 处理 latent patch tokens 替换 U-Net，并在 ImageNet class-conditional（256/512）上取得强结果。

**核心贡献**：
- FID 与计算量（Gflops）呈稳定改善关系，体现 Transformer 式 scaling 特征
- 建立了"latent patch tokenization + AdaLN-Zero 调制 + ViT 式堆叠"的可扩展配方
- 官方实现：[facebookresearch/DiT](https://github.com/facebookresearch/DiT)

**后续重要衍生**：
- **SD3 / FLUX**：文生图，引入 MM-DiT（多模态双流注意力）
- **CogVideoX**：视频生成，将 DiT 扩展到时空维度
- **Sora**（推测）：视频生成，基于视频 patch（spacetime patch）的 DiT 变体

---

### 2.2 DiT 最小架构分解

以经典 DiT 为参考，可抽象为五个可复用模块：

#### 模块一：Latent Patchify（tokenizer）

将 VAE latent（`B × C × H × W`）切成 `P × P` patch，展平成 token 序列，再线性映射到隐藏维度 `D`。

- patch size 直接影响 token 数与计算量：patch 越小 token 越多，Gflops 增长明显
- 常见配置：patch_size ∈ {2, 4, 8}

#### 模块二：Timestep Embedding（扩散步嵌入）

对扩散时间步 `t` 做正弦嵌入 + MLP，得到 `D` 维 conditioning 向量。

```python
import torch
import torch.nn as nn
import math

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = timestep_embedding(t, self.freq_embed_size)
        return self.mlp(x)
```

#### 模块三：Condition Embedding（类别/文本条件）

DiT 论文中主要是 class embedding；实际文生图可把文本编码后通过 cross-attn 或 AdaLN 调制注入。

```python
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids=None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)
```

#### 模块四：DiT Block（Transformer block + AdaLN-Zero 调制）

**AdaLN-Zero** 是 DiT 的关键：把时间步与条件信号注入到每个 block 的归一化/残差路径，比"把条件向量加到 token 上"更稳定、可控。

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        h = self.modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h

        h = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x
```

#### 模块五：Output Head（回归噪声/速度/数据）

```python
class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.norm_final(x)
        return self.linear(x)
```

---

### 2.3 完整 DiT 骨架（最小可运行示例）

```python
import torch
import torch.nn as nn
import math
from einops import rearrange


class DiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.zeros_(self.x_embedder.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2).flatten(2)
        return x

    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, _ = x.shape
        p = self.patch_size
        c = self.in_channels
        h, w = H // p, W // p
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, c, H, W)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.patchify(x)
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x, H, W)


def dit_s(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

def dit_b(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def dit_l(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def dit_xl(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


if __name__ == "__main__":
    model = dit_b(input_size=32, patch_size=2, in_channels=4, num_classes=1000)
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 1000, (2,))
    out = model(x, t, y)
    print(f"input: {x.shape} -> output: {out.shape}")
```

---

### 2.4 DiT 训练与采样流程

#### 训练（DDPM 风格 ε-prediction）

```python
import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int = 1000):
    beta_start, beta_end = 0.0001, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMScheduler:
    def __init__(self, timesteps: int = 1000):
        self.T = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        ac = self.alphas_cumprod.to(x0.device)[t]
        ac = ac.view(-1, 1, 1, 1)
        return ac.sqrt() * x0 + (1 - ac).sqrt() * noise, noise

    def train_step(self, model, x0: torch.Tensor, y: torch.Tensor, optimizer):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t, y)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
```

#### 采样（DDPM 基础版）

```python
@torch.no_grad()
def ddpm_sample(model, scheduler: DDPMScheduler, shape, y, device="cuda"):
    x = torch.randn(shape, device=device)
    betas = linear_beta_schedule(scheduler.T).to(device)
    alphas = 1.0 - betas
    ac = scheduler.alphas_cumprod.to(device)

    for t_val in reversed(range(scheduler.T)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t, y)
        alpha_t = alphas[t_val]
        ac_t = ac[t_val]
        x0_pred = (x - (1 - ac_t).sqrt() * pred_noise) / ac_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        if t_val > 0:
            ac_prev = ac[t_val - 1]
            beta_t = betas[t_val]
            mean = (ac_prev.sqrt() * beta_t * x0_pred + alpha_t.sqrt() * (1 - ac_prev) * x) / (1 - ac_t)
            var = beta_t * (1 - ac_prev) / (1 - ac_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0_pred
    return x
```

#### Classifier-Free Guidance（CFG）

```python
@torch.no_grad()
def cfg_sample(model, scheduler, shape, y, guidance_scale: float = 4.0, device="cuda"):
    x = torch.randn(shape, device=device)
    y_null = torch.full_like(y, model.y_embedder.num_classes)
    betas = linear_beta_schedule(scheduler.T).to(device)
    alphas = 1.0 - betas
    ac = scheduler.alphas_cumprod.to(device)

    for t_val in reversed(range(scheduler.T)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        noise_cond = model(x, t, y)
        noise_uncond = model(x, t, y_null)
        noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        alpha_t = alphas[t_val]
        ac_t = ac[t_val]
        x0_pred = (x - (1 - ac_t).sqrt() * noise) / ac_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        if t_val > 0:
            ac_prev = ac[t_val - 1]
            beta_t = betas[t_val]
            mean = (ac_prev.sqrt() * beta_t * x0_pred + alpha_t.sqrt() * (1 - ac_prev) * x) / (1 - ac_t)
            var = beta_t * (1 - ac_prev) / (1 - ac_t)
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = x0_pred
    return x
```

---

## 3. 典型模型/路线对比

### 3.1 注意力/长上下文/推理效率对比

| 方向 | 代表方法 | 关键思想 | 复杂度/收益侧重点 | 适用场景 |
|---|---|---|---|---|
| 精确注意力加速（kernel） | FlashAttention 1/2/3 | IO-aware + 并行划分 + 低精度 | 不改变模型，赢在吞吐/显存 | 训练与推理通用（GPU 友好） |
| 稀疏注意力 | Longformer | 局部窗口 + 少量全局 token | 近线性，需改注意力模式 | 长文档 NLP |
| 近似线性注意力 | Performer | 核方法随机特征近似 softmax | 线性复杂度，引入近似误差 | 超长序列、可容忍近似 |
| KV cache 降开销（结构） | GQA | 减少 KV head 数 | 推理带宽/显存显著下降 | LLM 解码高吞吐 |
| Serving 内存管理（系统） | PagedAttention/vLLM | KV cache 分页分配与共享 | 大幅减少碎片与浪费 | 多请求并发服务 |

### 3.2 DiT vs U-Net 扩散骨干对比

| 维度 | U-Net 扩散（传统） | DiT（Transformer 骨干） |
|---|---|---|
| 归纳偏置 | 强局部性、多尺度卷积 | token 化后主要靠注意力/MLP 学习 |
| 扩展方式 | 通常靠加宽/加深 + 多尺度设计 | 类似 ViT/LLM：深度/宽度/token 数都可扩展 |
| 条件注入 | 常见：FiLM/条件残差/交叉注意力 | 常见：AdaLN-Zero 在每层调制 |
| 计算热点 | 卷积 + 注意力（若有） | 注意力与 MLP（更利于 fused kernel/编译优化） |
| Scaling 行为 | 受卷积归纳偏置限制 | FID 随 Gflops 稳定改善 |
| 工程生态 | 成熟（Stable Diffusion 等） | 近年快速成熟（官方代码与复现增多） |

### 3.3 DiT 变体对比

| 变体 | 条件注入方式 | 主要改进 | 代表应用 |
|---|---|---|---|
| DiT（原版） | AdaLN-Zero | 建立基础配方 | ImageNet class-cond |
| SD3 / MM-DiT | 双流注意力 | 文本-图像双流，交叉更充分 | 文生图 |
| FLUX | MM-DiT + 单流混合 | 更高分辨率，多宽高比 | 文生图 |
| CogVideoX | 3D DiT（时空 patch） | 视频生成 | 文生视频 |
| Sora（推测） | 时空 patch DiT | 超长视频，可变时空分辨率 | 文生视频 |

---

## 4. 可复用 PyTorch 工程骨架

### 4.1 目录结构（推荐按四层分离）

```
project/
  pyproject.toml
  configs/
    dit_min.yaml
  src/
    models/
      transformer/
        attention.py        # SDPA wrapper / GQA(可选) / mask utils
        blocks.py           # PreNorm block, MLP, RMSNorm(可选)
      dit/
        dit.py              # DiT backbone (patchify, blocks, head)
        embeddings.py       # timestep emb, label emb
    diffusion/
      schedules.py          # beta/alpha schedule
      ddpm.py               # q_sample, p_sample (baseline)
      solvers.py            # DPM-Solver(可选扩展)
      guidance.py           # CFG
    train/
      loop.py               # train step, EMA(可选)
    utils/
      checkpoint.py
      seed.py
  scripts/
    train_min.py
    sample_min.py
```

**四层分离原则**：
1. **算子后端层**：`attention.py` 统一走 SDPA，屏蔽 FlashAttention 版本差异
2. **模型层**：DiT block、embedding 等纯 `nn.Module`，无训练/采样逻辑
3. **扩散/训练策略层**：schedule、solver、guidance 作为可插拔策略
4. **运行入口层**：`train_min.py` / `sample_min.py`，避免研究迭代把代码搅在一起

### 4.2 SDPA Wrapper（注意力统一后端）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
```

### 4.3 关键实现约定

```python
import torch

TORCH_COMPILE_ENABLED = False

def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if TORCH_COMPILE_ENABLED:
        return torch.compile(model)
    return model
```

**统一约定**：
- 注意力统一走 `F.scaled_dot_product_attention`，框架自动选择 fused kernel
- 推理/采样入口支持 `torch.compile`（可选开关），减少图断裂与 kernel fusion
- token 序列一律 `[B, N, D]`；latent 一律 `[B, C, H, W]`；避免在 block 内部频繁 permute

---

## 5. 与 Tri-Transformer 的关联

| DiT 组件 | Tri-Transformer 中的对应 | 说明 |
|---|---|---|
| AdaLN-Zero 调制 | **C-Transformer**（控制中枢） | adaLN-Zero 实现硬性可控性，详见 [05_adaln_zero.md](./05_adaln_zero.md) |
| Latent patch tokenization | 多模态 tokenizer | VQ-GAN/EnCodec/SNAC 各模态离散化后拼接 |
| Timestep embedding | 扩散噪声控制 | C-Transformer 生成控制中接入噪声尺度信号 |
| CFG guidance | 实时可控生成 | 无条件/有条件联合训练，推理时线性组合 |
| DiT Block（ViT 式堆叠） | I/C/O-Transformer Block | 三分支各自独立堆叠，通过 state slots 交互 |
| Scaling（FID vs Gflops） | 可持续迭代路线 | Transformer 骨干的 scaling 特性是选择 DiT 的核心理由 |

---

## 参考

| 方法 | 论文/链接 |
|---|---|
| Reformer | Kitaev et al., 2020. *Reformer: The Efficient Transformer* |
| Longformer | Beltagy et al., 2020. *Longformer: The Long-Document Transformer* |
| Performer | Choromanski et al., 2020. *Rethinking Attention with Performers* |
| FlashAttention | Dao et al., 2022. arXiv:2205.14135 |
| FlashAttention-2 | Dao, 2023. arXiv:2307.08691 |
| FlashAttention-3 | Shah et al., 2024. arXiv:2407.08608 |
| RoPE | Su et al., 2021. arXiv:2104.09864 |
| ALiBi | Press et al., 2022. arXiv:2108.12409 |
| YaRN | Peng et al., 2023. arXiv:2309.00071 |
| GQA | Ainslie et al., 2023. arXiv:2305.13245 |
| PagedAttention/vLLM | Kwon et al., 2023. arXiv:2309.06180 |
| RetNet | Sun et al., 2023. arXiv:2307.08621 |
| Mamba | Gu & Dao, 2023. arXiv:2312.00752 |
| DiT | Peebles & Xie, 2022. arXiv:2212.09748 |
| DPM-Solver | Lu et al., 2022. arXiv:2206.00927 |
| CFG | Ho & Salimans, 2022. arXiv:2207.12598 |
| EDM | Karras et al., 2022. arXiv:2206.00364 |
| SD3 / MM-DiT | Esser et al., 2024. arXiv:2403.03206 |
