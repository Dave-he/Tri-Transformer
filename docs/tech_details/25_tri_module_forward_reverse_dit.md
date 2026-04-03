# 反向Transformer、正向Transformer与DiT联合架构
## 连接拓扑、对齐与训练范式（2022–至今）

## 0. 结论先行

三模块联合架构的核心工程建议：

1. **能跑、稳、易迭代**：选串联拓扑A（Forward-T → DiT/Reverse-T）+ AdaLN-Zero；先训DiT至强去噪，再训Forward-T，最后小步联合微调。
2. **理论上更像真实双向过程**：选IPF/DSB式交替拓扑E，代价是工程复杂度显著上升。
3. **参数/算力最省**：选共享干线双头拓扑C，但必须加强双向一致性约束与EMA稳定技巧。

**最关键工程约定**：三模块必须共享同一噪声调度 `α̅_t`、同一预测参数化（`ε`或`v`）、同一输入缩放，否则Forward-T输出对DiT/Reverse-T是分布外输入，导致采样崩坏。

---

## 1. 三模块定义与职能

| 模块 | 职能 | 典型参数化 |
|---|---|---|
| **DiT（Diffusion Transformer）** | 反向扩散去噪主干（denoiser backbone），在VAE latent patch tokens上做Transformer去噪 | ε/v/x₀ prediction，adaLN-Zero条件注入 |
| **反向Transformer（Reverse-T）** | 学习reverse-time dynamics（噪声→数据），工程上可就是DiT，也可以是额外refiner头 | 与DiT等价或作为轻量强化头 |
| **正向Transformer（Forward-T）** | 学习forward-time dynamics，承担三种职能之一（见下） | velocity/ε_fwd/Δx/z_t* |

**Forward-T的三种可选职能**（选其一，决定loss与采样闭环）：

- **职能A（可学习前向SDE）**：不再固定高斯加噪，走向Schrödinger Bridge / IPF类"双向过程"
- **职能B（inversion/encoding）**：把真实样本映射到可控噪声/latent状态，用于编辑、对齐或蒸馏
- **职能C（flow速度场）**：与SiT / Rectified Flow / Interpolant框架兼容，预测连续时间velocity

---

## 2. 串并联连接拓扑（MECE分类）

### 2.1 串联：Forward-T → DiT/Reverse-T

**拓扑A（最常用，工程最稳）**：Forward-T先对齐到合适噪声域，DiT负责高保真生成/重建。

```
x0 --(VAE Enc)--> z0
z0 --(Forward-T: encode/invert)--> z_t*  (或 ε_t*, v_t*, 轨迹参数)
z_t* --(DiT/Reverse-T denoise)--> z0_hat --(VAE Dec)--> x_hat
```

适用场景：编辑/条件重建/分布迁移；Forward-T提供语义对齐的初值，DiT负责细节保真。

**风险**：Forward-T若产生分布外 `z_t*`，会导致DiT采样崩坏（需加分布约束/对齐损失，见第5节）。

**拓扑B（粗细两段）**：Forward-T粗去噪，DiT精refinement，适合把算力集中在最后若干步。

```
z_T -> Forward-T (coarse, few-step) -> z_k -> DiT/Reverse-T (many-step) -> z0
```

---

### 2.2 并联：Forward-T ∥ DiT/Reverse-T

**拓扑C（共享干线双头，参数最省）**：同一Trunk，两套direction head。

```
h_t = Trunk(tokens(z_t), cond, t)
ε_rev = Head_rev(h_t)      # 反向采样
ε_fwd = Head_fwd(h_t)      # 正向/bridge/IPF
```

Trunk直接用DiT block堆叠（patchify + Transformer blocks + unpatchify），Head可很轻（1–2层MLP），把双向差异留给head，减少算力开销。

**拓扑D（双塔独立，稳定性最高）**：Forward-T与DiT各自独立，共享Condition Encoder。

```
cond_enc: y -> c
Forward-T(z_t, c, t) -> ε_fwd
DiT(z_t, c, t)       -> ε_rev
```

优点：互不干扰，最易稳定；缺点：参数与算力翻倍（研发排障/对照实验首选）。

---

### 2.3 交织/迭代：IPF/Schrödinger Bridge式

**拓扑E（IPF交替训练，理论最正）**：Forward-T与DiT交替拟合边缘分布，实现DSB（Diffusion Schrödinger Bridge）思想。

```
第n轮：  固定DiT(backward)  → 训练Forward-T，使forward轨迹末端匹配prior/噪声边缘
第n+1轮：固定Forward-T      → 训练DiT(backward)，使末端匹配数据边缘
```

优点：理论上逼近真实双向桥接分布；缺点：工程复杂，训练周期更长。

---

### 2.4 拓扑选择决策表

| 拓扑 | 稳定性 | 训练难度 | 推理算力 | 推荐场景 |
|---|---|---|---|---|
| A 串联（Forward→DiT） | 高 | 低 | 低 | **首选**：编辑/重建/快速迭代 |
| B 粗细串联 | 高 | 低 | 中 | 算力集中在refine阶段 |
| C 共享干线双头 | 中 | 高 | 低-中 | 参数/算力受限时 |
| D 双塔独立 | 最高 | 中 | 高 | 研发排障/对照实验 |
| E IPF交替 | 中 | 最高 | 中 | 理论研究/桥接分布建模 |

---

## 3. 共享/独立编码器与条件注入

### 3.1 编码器共享策略

| 方案 | 结构 | 何时用 | 主要风险 |
|---|---|---|---|
| **S1 共享Condition Encoder** | cond_enc产出c，供三模块共用 | 文本/多模态条件很重（CLIP/T5） | 条件分布漂移会同时伤三路 |
| **S2 共享Patchify/Tokenizer** | 共享VAE+patchify投影 | latent空间统一 | 共享过多会耦合训练不稳 |
| **S3 完全独立** | 各自encoder/patchify | 研究原型、排查不稳定 | 参数与显存翻倍 |

**实践建议**：共享Condition Encoder + 共享VAE latent，Transformer blocks按稳定性选择。

### 3.2 条件注入方式对比

| 注入方式 | 计算开销 | 控制力 | 推荐程度 |
|---|---|---|---|
| **adaLN-Zero** | 几乎0 Gflops额外 | 高（timestep + 全局条件） | **首选**，残差门控零初始化确保稳定 |
| cross-attention | +~15% Gflops | 最强（text token序列） | 文本条件强控制时选用 |
| in-context token拼接 | 几乎0 Gflops | 中（依赖token交互） | 简单实现备选 |

**推荐配置**：
- timestep + 全局条件（class/style/camera）→ AdaLN-Zero
- 序列文本条件 → cross-attention（控制强但慢）或池化为单向量后走AdaLN-Zero（快但控制弱）

---

## 4. token/latent/时间轴对齐（避免接口不闭合）

### 4.1 latent与token维度统一

三模块以同一latent张量形状 `(B, C, H, W)` 作为主接口，再各自patchify，避免token级对齐难题。

| 分辨率 | VAE latent形状 | patch_size=2 token数 |
|---|---|---|
| 256×256 | 32×32×4 | 256 |
| 512×512 | 64×64×4 | 1024 |

### 4.2 时间参数t对齐

- **离散DDPM（最稳）**：三模块统一用 T=1000 线性方差日程，同一组 `α̅_t`，输出同参数化（`ε` 或 `v`）
- **连续时间（SiT/Rectified Flow/Interpolant）**：三模块统一 t∈[0,1]，在需要兼容扩散调度器处做映射 `t → α̅_t` 或 `σ(t)`

### 4.3 噪声调度与参数化对齐（三条硬规则）

1. **同一噪声族**：Forward-T和DiT必须共享同一 `σ(t)` / `α̅_t` 定义，否则Forward-T给出的 `z_t` 对DiT是分布外输入
2. **同一预测参数化**：统一 `ε`-prediction 或 `v`-prediction，不可混用
3. **同一输入缩放/预条件**：若采用EDM式预条件需同步到Forward-T，否则双向一致性损失会学到"缩放补丁"而非动力学

---

## 5. 训练策略：Loss组合与稳定性

### 5.1 推荐训练路线（从稳到激进）

**路线R1（最稳：两阶段 + 最后小步联合）**：
1. 先训DiT/Reverse-T：标准latent扩散训练，AdaLN-Zero注入(t,c)，MSE噪声损失
2. 冻结DiT，训Forward-T：监督信号为"被DiT能还原"
3. 小学习率联合微调：只微调Forward-T与条件相关层（或DiT的modulation MLP）

**路线R2（并联共享干线）**：直接共享Trunk，双头分别做forward/backward预测（拓扑C），需更强正则。

**路线R3（IPF/DSB式交替）**：交替更新forward与backward过程，逐步逼近桥接分布，工程复杂度最高。

### 5.2 Loss配方

```python
# (1) 反向去噪主损失（必须）
L_rev = MSE(ε_hat_rev, ε)

# (2) 正向一致性损失（按Forward-T职能选一种）
# 若Forward-T输出 z_t*：
L_cycle = MSE(Denoise_DiT(z_t*, t*), z0)
# 若Forward-T输出 ε_fwd：
L_fwd = MSE(ε_hat_fwd, ε)

# (3) 双向对偶一致性正则（并联/共享干线时强烈建议）
# 把 ε_hat 换算成 x0_hat 或 v_hat 后对齐，减少两头学成两套坐标系

# (4) 总loss
L = L_rev + λ_fwd * L_fwd + λ_cycle * L_cycle
```

### 5.3 稳定性技巧

| 技巧 | 说明 | 重要性 |
|---|---|---|
| **EMA** | DiT训练广泛使用EMA权重稳定采样与指标 | 必须 |
| **AdaLN-Zero初始化** | 每个block初始近似identity，显著改善稳定性 | 必须 |
| **冻结/解冻策略** | 联合微调时先冻住DiT再逐层解冻 | 强烈建议 |
| **head轻量化** | 拓扑C中head用1–2层MLP，把双向差异留给head | 建议 |
| **分布约束** | Forward-T输出 z_t* 需加约束防止分布外输入 | 拓扑A必须 |

---

## 6. PyTorch工程骨架

### 6.1 接口约定

```python
z_t: (B, C=4, H, W)    # LDM latent，三模块统一接口
c:   (B, D_cond)        # 共享 cond_enc 输出
t:   int64 离散步 或 float 连续时间（三模块必须一致）
```

### 6.2 共享条件编码器

```python
import torch
import torch.nn as nn


class CondEncoder(nn.Module):
    def __init__(self, cond_dim_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1024, cond_dim_out),
            nn.SiLU(),
            nn.Linear(cond_dim_out, cond_dim_out),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.proj(y)
```

### 6.3 正向Transformer（Forward-T）

```python
class ForwardTransformer(nn.Module):
    def __init__(self, trunk: nn.Module, out_channels: int = 4):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Conv2d(trunk.out_channels, out_channels, 1)

    def forward(
        self,
        z_t: torch.Tensor,
        t_idx: torch.Tensor,
        c_vec: torch.Tensor,
    ) -> torch.Tensor:
        h = self.trunk(z_t, t_idx, c_vec)
        return self.head(h)
```

### 6.4 三模块联合系统（串联/并联可切换）

```python
import torch.nn.functional as F


def make_noisy(
    z0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    ac = alphas_cumprod.to(z0.device)[t].view(-1, 1, 1, 1)
    return ac.sqrt() * z0 + (1 - ac).sqrt() * noise


class TriModuleSystem(nn.Module):
    def __init__(
        self,
        cond_enc: nn.Module,
        forward_T: nn.Module,
        reverse_T: nn.Module,
        alphas_cumprod: torch.Tensor,
    ):
        super().__init__()
        self.cond_enc = cond_enc
        self.forward_T = forward_T
        self.reverse_T = reverse_T
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(
        self,
        z0: torch.Tensor,
        t_idx: torch.Tensor,
        noise: torch.Tensor,
        y_cond: torch.Tensor,
        mode: str = "parallel",
    ):
        c = self.cond_enc(y_cond)
        z_t = make_noisy(z0, t_idx, noise, self.alphas_cumprod)

        if mode == "serial":
            z_t_star = self.forward_T(z0, t_idx, c)
            eps_rev = self.reverse_T(z_t, t_idx, y_cond)
            return eps_rev, z_t_star

        elif mode == "parallel":
            eps_rev = self.reverse_T(z_t, t_idx, y_cond)
            eps_fwd = self.forward_T(z_t, t_idx, c)
            return eps_rev, eps_fwd

        else:
            raise ValueError(f"Unknown mode: {mode}")
```

### 6.5 训练循环（并联模式）

```python
import torch
from torch.optim import AdamW
from copy import deepcopy


def train_step(
    model: TriModuleSystem,
    z0: torch.Tensor,
    y: torch.Tensor,
    optimizer: AdamW,
    ema_model: TriModuleSystem,
    lam_fwd: float = 0.5,
    ema_decay: float = 0.9999,
):
    t = torch.randint(0, 1000, (z0.shape[0],), device=z0.device)
    noise = torch.randn_like(z0)

    eps_rev, eps_fwd = model(z0, t, noise, y, mode="parallel")

    loss_rev = F.mse_loss(eps_rev, noise)
    loss_fwd = F.mse_loss(eps_fwd, noise)
    loss = loss_rev + lam_fwd * loss_fwd

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

    return {"loss_rev": loss_rev.item(), "loss_fwd": loss_fwd.item()}
```

---

## 7. 与Tri-Transformer架构的关联

| 联合架构概念 | Tri-Transformer对应模块 | 说明 |
|---|---|---|
| DiT / Reverse-T | **C-Transformer**（DiT控制中枢） | adaLN-Zero注入t/c，负责高保真去噪，详见 [04_dit.md](./04_dit.md)、[05_adaln_zero.md](./05_adaln_zero.md) |
| Forward-T（inversion/encoding） | **I-Transformer**（正向实时编码） | 把输入映射到latent/噪声域，实现"可控编辑"入口 |
| 共享Condition Encoder | C-Transformer条件注入层 | 文本/类别条件统一编码，供三分支共用 |
| 串联拓扑A | I→C→O三阶段信息流 | I-Transformer编码 → C-Transformer控制 → O-Transformer输出，通过State Slots传递中间态 |
| AdaLN-Zero条件注入 | C-Transformer block内调制 | 详见 [05_adaln_zero.md](./05_adaln_zero.md)，用于硬性可控性 |
| EMA稳定训练 | 三分支分阶段缝合方案 | 详见 [19_lora_qlora.md](./19_lora_qlora.md) 中的分阶段微调策略 |

---

## 8. 扩展方向

### 8.1 与Rectified Flow / SiT的兼容

若Forward-T承担**flow速度场**职能，可与连续时间框架直接对接：

```python
# SiT/Rectified Flow参数化：预测速度 v = x1 - x0
# 训练目标：v_pred(x_t, t) ≈ x1 - x0，其中 x_t = (1-t)*x0 + t*x1
loss_flow = F.mse_loss(forward_T(x_t, t, c), x1 - x0)
```

### 8.2 少步生成/蒸馏

- **Consistency Models**思路：把DiT多步teacher压到少步student，减少采样步数
- **Progressive Distillation**：逐步减半采样步数（1000步→500→250→...→4步）
- 两者均可以Forward-T为"快速粗估模块"，DiT为"精细refine模块"

### 8.3 EDM式预条件的适配

若采用EDM预条件（`c_skip, c_out, c_in, c_noise`缩放），需对Forward-T同步处理：

```python
# EDM预条件（需三模块同步）
sigma = get_sigma(t)
c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
c_out  = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
c_in   = 1 / (sigma**2 + sigma_data**2).sqrt()
c_noise = sigma.log() / 4

x_input = c_in * x_noisy
out = model(x_input, c_noise, c)
D_theta = c_skip * x_noisy + c_out * out
```

---

## 参考

| 方法 | 论文/链接 |
|---|---|
| DiT | Peebles & Xie, 2022. *Scalable Diffusion Models with Transformers*. arXiv:2212.09748 |
| LDM | Rombach et al., 2022. *High-Resolution Image Synthesis with Latent Diffusion Models*. arXiv:2112.10752 |
| SiT | Ma et al., 2024. *Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers*. arXiv:2401.08740 |
| Rectified Flow | Liu et al., 2022. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv:2209.03003 |
| Schrödinger Bridge / DSB | De Bortoli et al., 2021. *Diffusion Schrödinger Bridge*. arXiv:2106.01357 |
| EDM | Karras et al., 2022. *Elucidating the Design Space of Diffusion-Based Generative Models*. arXiv:2206.00364 |
| Consistency Models | Song et al., 2023. *Consistency Models*. arXiv:2303.01469 |
| Progressive Distillation | Salimans & Ho, 2022. *Progressive Distillation for Fast Sampling of Diffusion Models*. arXiv:2202.00512 |
| U-ViT | Bao et al., 2022. *All are Worth Words: A ViT Backbone for Diffusion Models*. arXiv:2209.12152 |
| PixArt-α | Chen et al., 2023. *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*. arXiv:2310.00426 |
| MM-DiT / SD3 | Esser et al., 2024. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. arXiv:2403.03206 |
