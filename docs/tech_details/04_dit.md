# DiT（Diffusion Transformer）

## 0. 结论先行

- **核心贡献**：用 ViT 式 Transformer 替换扩散模型 U-Net 主干，在 ImageNet class-conditional 上建立"FID 随 Gflops 稳定改善"的规模律，是可扩展生成建模的标准骨干配方。
- **关键设计**：latent patch tokenization（VAE latent → patch tokens）+ adaLN-Zero 条件注入（时间步 + 类别/文本），其中 adaLN-Zero 零初始化让每个 block 初始为近似 identity，是训练稳定性的核心来源。
- **工程推荐**：条件注入优先选 adaLN-Zero（计算开销几乎为零）；文本序列条件可加 cross-attention 层（+~15% Gflops）；扩散 backbone 推荐 SDPA 统一后端，自动落到 FlashAttention fused kernel。
- **后续主流衍生**：SD3/MM-DiT（双流文本-图像注意力）、FLUX（混合单/双流）、CogVideoX（时空 3D patch）均基于此配方演化，规模律持续成立（模型越大 FID 越低，无明显天花板）。
- **Tri-Transformer 中的角色**：C-Transformer 借鉴 DiT 的条件调制机制（adaLN-Zero），以 DiT 架构作为"生成式控制中枢"，维护全局对话状态，通过条件调制控制 I/O 两端特征分布。

---

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

---

## 6. SD3 / FLUX 架构深度解析（与 Tri-Transformer 控制层的关联）

### 6.1 Stable Diffusion 3 的 MM-DiT 架构

SD3（arXiv:2403.03206，Stability AI 2024）将 DiT 推广为**多模态 DiT（MM-DiT）**，文本与图像 Token 在同一 Transformer 中交互，而非文本仅作为条件向量：

```python
class MMDiTBlock(nn.Module):
    """
    SD3 的多模态 DiT Block：文本 Token 与图像 Token 共同参与自注意力
    C-Transformer 可借鉴此设计实现对话控制向量与内容 Token 的深度融合
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_c = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.adaLN_x = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        self.adaLN_c = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_x[-1].weight)
        nn.init.zeros_(self.adaLN_c[-1].weight)
        nn.init.zeros_(self.adaLN_c[-1].bias)
        nn.init.zeros_(self.adaLN_x[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                timestep_emb: torch.Tensor) -> tuple:
        """
        x: [B, Tx, D] 图像/内容 Token
        c: [B, Tc, D] 控制/条件 Token（对话 C-Transformer 输出）
        timestep_emb: [B, D] 时间步嵌入（Tri-Transformer 中可替换为对话状态嵌入）
        """
        sx, sc = timestep_emb, timestep_emb

        px = self.adaLN_x(sx)
        shift_x, scale_x, gate_x, shift_xm, scale_xm, gate_xm = px.chunk(6, dim=-1)

        pc = self.adaLN_c(sc)
        shift_c, scale_c, gate_c, shift_cm, scale_cm, gate_cm = pc.chunk(6, dim=-1)

        x_mod = self.norm_x(x) * (1 + scale_x[:, None]) + shift_x[:, None]
        c_mod = self.norm_c(c) * (1 + scale_c[:, None]) + shift_c[:, None]

        xc = torch.cat([x_mod, c_mod], dim=1)
        attn_out, _ = self.attn(xc, xc, xc)
        attn_x, attn_c = attn_out[:, :x.size(1)], attn_out[:, x.size(1):]

        x = x + gate_x[:, None] * attn_x
        c = c + gate_c[:, None] * attn_c
        return x, c
```

**对 Tri-Transformer C-Transformer 的启示**：MM-DiT 证明条件 Token 与内容 Token 可以在同一注意力层中双向交互，远比单向的 adaLN 调制更强大，适合作为 C-Transformer 高阶版本的架构参考。

### 6.2 FLUX.1 的架构创新（Black Forest Labs, 2024）

FLUX.1（基于 SD3 演进）在 MM-DiT 基础上引入**双流与单流交替层**：

```
双流阶段（前 N/2 层）：
  图像 Token ─→ MMDiTBlock ←─ 文本/控制 Token
  （两者独立线性层，共享注意力）

单流阶段（后 N/2 层）：
  [图像 Token + 文本 Token 拼接] ─→ 标准 DiT Block
  （完全融合处理）
```

| 模型 | 参数量 | 特点 | 许可证 |
|---|---|---|---|
| FLUX.1-dev | 12B | 最高质量，非商业 | FLUX-1-dev 许可 |
| FLUX.1-schnell | 12B | 4步蒸馏，极快 | Apache 2.0 |
| FLUX.1-lite | 2.6B | 轻量版 | 商业友好 |

### 6.3 DiT 在视频生成中的时序扩展（CogVideoX, 2024）

CogVideoX（arXiv:2408.06072，智谱 AI 2024）将 DiT 扩展至视频：

```python
class CogVideoXBlock(nn.Module):
    """
    3D 时空 DiT Block：同时处理空间和时序 Token
    O-Transformer 的视频生成层可参考此设计
    """
    def __init__(self, d_model: int, num_heads: int, temporal_len: int):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        self.temporal_len = temporal_len

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T*H*W, D] 时空展平 Token
        c: [B, D] 条件向量
        """
        B, THW, D = x.shape
        T = self.temporal_len
        HW = THW // T

        shift, scale, gate, shift_t, scale_t, gate_t = \
            self.adaLN(c).chunk(6, dim=-1)

        x_s = x.reshape(B * T, HW, D)
        x_s_mod = x_s * (1 + scale[:, None]) + shift[:, None]
        x_s = x_s + gate[:, None] * self.spatial_attn(x_s_mod, x_s_mod, x_s_mod)[0]

        x_t = x_s.reshape(B, T, HW, D).permute(0, 2, 1, 3).reshape(B * HW, T, D)
        x_t_mod = x_t * (1 + scale_t[:, None]) + shift_t[:, None]
        x_t = x_t + gate_t[:, None] * self.temporal_attn(x_t_mod, x_t_mod, x_t_mod)[0]
        x = x_t.reshape(B, HW, T, D).permute(0, 2, 1, 3).reshape(B, THW, D)
        return x
```

### 6.4 DiT 规模律最新数据（2025）

| 模型 | GFLOPs | 参数量 | ImageNet FID | 发布时间 |
|---|---|---|---|---|
| DiT-XL/2（原始）| 118.6 | 675M | 2.27 | 2022 |
| SiT-XL/2 | 118.6 | 675M | 2.06 | 2024 |
| SD3-Medium | ~18T* | 2B | — | 2024 |
| FLUX.1-dev | ~100T* | 12B | — | 2024 |

*推理时总 FLOPs（含多步去噪）

**核心结论**：DiT 规模律持续成立——模型越大，质量越高，无明显天花板。这为 Tri-Transformer C-Transformer 选择更大的 DiT 骨干提供了充分理论支撑。
