# DiT（Diffusion Transformer）

## 1. 概述

Diffusion Transformer（DiT）是 William Peebles 与 Saining Xie 于 2022 年提出的扩散模型架构（arXiv:2212.09748，ICCV 2023 最佳论文候选）。其核心贡献是：**用 Transformer 替换扩散模型中传统的 U-Net 主干**，作用于 VAE 编码的潜在空间 Patch，并以清晰的规模律（Scaling Law）验证了 Transformer 在生成任务中的可扩展性。

**在 Tri-Transformer 中的角色**：C-Transformer 借鉴 DiT 的条件调制机制（adaLN-Zero），以 DiT 架构作为"生成式控制中枢"，维护全局对话状态，通过条件调制控制 I/O 两端的特征分布。

---

## 2. 核心架构

### 2.1 整体流程

```
输入图像 x
   ↓ VAE Encoder
潜在表征 z ∈ R^{H/8 × W/8 × C}
   ↓ Patchify（分 Patch，类 ViT）
Token 序列 + 位置编码
   ↓
[DiT Block × L 层]
   ↓ 条件信号注入（timestep t + class label y）
   ↓ 去噪预测 ε_θ(z_t, t, y)
   ↓ VAE Decoder
去噪后图像
```

### 2.2 DiT Block 结构

```python
class DiTBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] 输入 Token 序列
        c: [B, D]    条件嵌入（timestep + class 融合后的向量）
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

### 2.3 条件注入方式对比

DiT 论文比较了 4 种条件注入策略：

| 策略 | 原理 | FID（越低越好） | 本项目适用性 |
|---|---|---|---|
| In-context conditioning | 条件 Token 拼接到输入序列 | 较差 | 低（增加序列长度） |
| Cross-attention | 条件作为 KV，内容作为 Q | 中等 | 中 |
| Adaptive LayerNorm (adaLN) | 条件生成 scale/shift 调制 LN | 好 | 高 |
| **adaLN-Zero** | adaLN + gate 归零初始化 | **最好** | **最高** |

adaLN-Zero 是 DiT 中表现最佳的策略，也是本项目 C-Transformer 所采用的核心机制。

---

## 3. 规模律与模型变体

### 3.1 DiT 规模变体

| 模型 | 层数 | 隐藏维度 | 注意力头 | GFLOPs | FID-256 |
|---|---|---|---|---|---|
| DiT-S/2 | 12 | 384 | 6 | 6.1 | 68.4 |
| DiT-B/2 | 12 | 768 | 12 | 23.0 | 43.5 |
| DiT-L/2 | 24 | 1024 | 16 | 80.7 | 23.3 |
| DiT-XL/2 | 28 | 1152 | 16 | 118.6 | **2.27** |

**规模律结论**：GFLOPs（计算量）与 FID 呈强负相关，即更大的模型始终带来更好的生成质量。这与语言模型的规模律一致，验证了 Transformer 在生成任务中的可扩展性。

### 3.2 Patch 大小的影响

Patch 越小，序列越长，模型计算量越大，质量越高：
- `/2`：Patch size 2，序列长度 256（对应 32×32 latent）
- `/4`：Patch size 4，序列长度 64
- `/8`：Patch size 8，序列长度 16

---

## 4. 使用方法

### 4.1 官方实现快速上手

```python
git clone https://github.com/facebookresearch/DiT
cd DiT
pip install -r requirements.txt

python sample.py --model DiT-XL/2 \
    --image-size 256 \
    --num-classes 1000 \
    --cfg-scale 4.0 \
    --ckpt /path/to/DiT-XL-2-256x256.pt
```

### 4.2 作为 C-Transformer 控制中枢的适配

```python
class CTransformer(nn.Module):
    """
    基于 DiT 架构的对话控制中枢
    将图像扩散条件替换为对话控制条件
    """
    def __init__(self, d_model=1024, num_heads=16, num_layers=12, num_state_slots=8):
        super().__init__()
        self.state_slots = nn.Parameter(torch.randn(1, num_state_slots, d_model))
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn_i = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn_o = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.dit_blocks = nn.ModuleList([DiTBlock(d_model, num_heads) for _ in range(num_layers)])
        self.condition_proj = nn.Linear(d_model, d_model)

    def forward(self, i_enc, o_prev, external_condition=None):
        B = i_enc.size(0)
        s = self.state_slots.expand(B, -1, -1)
        s, _ = self.self_attn(s, s, s)
        s, _ = self.cross_attn_i(s, i_enc, i_enc)
        s, _ = self.cross_attn_o(s, o_prev, o_prev)

        c = s.mean(dim=1)
        if external_condition is not None:
            c = c + self.condition_proj(external_condition)

        for block in self.dit_blocks:
            s = block(s, c)

        return s, c
```

---

## 5. 最新进展（2024-2025）

### 5.1 SiT（Stochastic Interpolants）
- 将 DiT 架构应用于更一般的生成框架（非 DDPM），FID 进一步改善。

### 5.2 SD3 / FLUX（Stability AI, 2024）
- Stable Diffusion 3 与 FLUX 均基于 DiT 架构，引入"多模态 DiT（MM-DiT）"，文本与图像 Token 在同一 Transformer 中交互，生成质量大幅提升。

### 5.3 Open-Sora / CogVideoX（2024）
- 将 DiT 扩展至视频生成，引入 3D Patch 化（时空联合分块）与时序因果掩码，是 C-Transformer 扩展至视频控制的参考路径。

### 5.4 MAR（Masked AutoRegressive）
- 将 DiT 的条件调制机制与自回归生成结合，证明扩散特征（continuous diffusion head）可以叠加在 AR Transformer 上，为 Tri-Transformer 的混合生成范式提供参考。
