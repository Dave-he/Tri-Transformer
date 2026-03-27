# adaLN-Zero（自适应层归一化零初始化）

## 1. 概述

adaLN-Zero（Adaptive Layer Norm with Zero Initialization）是 DiT 论文中提出的条件调制机制，也是 Tri-Transformer C-Transformer 控制中枢的核心技术。它通过从条件信号中预测 `scale`、`shift`、`gate` 三个参数，以无侵入方式动态调制被控模块的特征分布，实现对生成过程的实时精确控制。

**在 Tri-Transformer 中的角色**：C-Transformer 通过 adaLN-Zero 向 I-Transformer 各层和 O-Transformer 各层注入控制信号，实现生成过程中途的风格、情感、语速等属性的实时切换。

---

## 2. 实现原理

### 2.1 完整机制

标准 LayerNorm：
$$y = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

其中 $\gamma$、$\beta$ 是可学习的固定参数。

adaLN（无零初始化版本）：
$$y = \frac{x - \mu}{\sigma} \cdot (1 + \text{scale}(c)) + \text{shift}(c)$$

**adaLN-Zero**（加入 gate，并零初始化线性层）：
$$x' = x + \text{gate}(c) \odot \left[\frac{x - \mu}{\sigma} \cdot (1 + \text{scale}(c)) + \text{shift}(c)\right]$$

其中 `gate`、`scale`、`shift` 均从条件向量 $c$ 通过线性层动态预测，且预测线性层**权重初始化为零**，使得训练初始状态每个 DiT Block 等价于恒等函数（Identity Function）。

### 2.2 PyTorch 实现

```python
import torch
import torch.nn as nn

class AdaLNZeroBlock(nn.Module):
    """
    带 adaLN-Zero 条件调制的 Transformer Block
    支持 Self-Attention 和 MLP 两个子模块的独立调制
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_ff, d_model)
        )
        self.adaLN_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_proj[-1].weight)
        nn.init.zeros_(self.adaLN_proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] 被调制的 Token 序列
        c: [B, D]    条件信号（来自 C-Transformer 状态槽）
        """
        params = self.adaLN_proj(c)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params.chunk(6, dim=-1)

        x_norm = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_a[:, None] * attn_out

        x_norm2 = self.norm2(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(x_norm2)
        return x
```

### 2.3 零初始化的意义

```python
nn.init.zeros_(self.adaLN_proj[-1].weight)
nn.init.zeros_(self.adaLN_proj[-1].bias)
```

训练开始时，由于输出层权重全为零：
- `scale = 0`：LayerNorm 输出不被缩放（等效标准 LN）。
- `shift = 0`：LayerNorm 输出不被偏移。
- `gate = 0`：**整个残差分支输出为 0**，Block 等价于恒等映射。

这保证了训练初始化的稳定性，使深层网络从"不做任何变换"开始逐渐学习控制信号，**避免了深层模型训练初期的梯度爆炸**。

---

## 3. 与其他条件注入方式的对比

### 3.1 加法注入（Simple Addition）
```python
x = x + condition_proj(c)
```
最简单，但条件作用单一（仅平移特征空间），无法精细调制尺度。

### 3.2 FiLM（Feature-wise Linear Modulation）
```python
gamma, beta = condition_proj(c).chunk(2, dim=-1)
x = gamma * x + beta
```
与 adaLN 类似，但作用在原始特征而非归一化后特征，数值稳定性较差。

### 3.3 Cross-Attention 注入
```python
x = x + CrossAttention(Q=x, KV=condition)
```
最灵活，条件可以是序列（而非单一向量），但计算代价高。

### 3.4 adaLN-Zero（推荐）
- 同时控制尺度（scale）、偏移（shift）和门控（gate）三个维度。
- 归一化后作用，数值稳定。
- 零初始化保证训练稳定。
- 在 DiT 实验中显著优于其他方式（FID 降低约 30%）。

---

## 4. 在多模态对话控制中的扩展

### 4.1 层级调制（Per-Layer Modulation）

C-Transformer 为 I/O 每一层分别生成独立的调制参数：

```python
class CTransformerModulator(nn.Module):
    def __init__(self, d_model: int, num_io_layers: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 6 * d_model * num_io_layers)

    def forward(self, state_slot: torch.Tensor):
        """
        state_slot: [B, D]
        返回: 每一层的 (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m)
        """
        all_params = self.proj(state_slot)
        return all_params.view(state_slot.size(0), -1, 6, state_slot.size(-1))
```

### 4.2 实时控制演示

```python
state_slot_normal = c_transformer.get_state("正常语速")
state_slot_urgent = c_transformer.get_state("紧急语气，加快语速")

for token in audio_stream:
    if time > interrupt_point:
        ctrl = compute_adaln_params(state_slot_urgent)
    else:
        ctrl = compute_adaln_params(state_slot_normal)

    output_token = o_decoder.step(token, ctrl_scale=ctrl.scale, ctrl_shift=ctrl.shift)
```

---

## 5. 最新进展（2024-2025）

### 5.1 adaLN 在视频生成中的应用
- **Open-Sora**、**CogVideoX**：将 adaLN-Zero 扩展至时序维度，使用时间步（Timestep）+帧序号（Frame Index）作为条件，实现对视频生成的逐帧精确控制。

### 5.2 Lumina-T2X（2024）
- 将 adaLN 调制推广至文本-图像/视频/音频统一生成，使用同一控制信号驱动多模态输出。

### 5.3 Diffusion Forcing（2024）
- 将扩散过程的逐步去噪与自回归生成结合，adaLN 调制信号由噪声水平（noise level）替代时间步，为序列生成提供更细粒度的控制。这与 Tri-Transformer C-Transformer 的对话状态控制具有深刻的类比关系。

### 5.4 直接偏好优化（DPO）与调制
- 最新研究探索将 RLHF/DPO 训练的偏好信号通过 adaLN 实时注入，允许在推理时动态调整模型输出的风格与安全性，无需重新训练。
