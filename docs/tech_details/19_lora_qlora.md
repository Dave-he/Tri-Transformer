# LoRA & QLoRA（低秩适配微调）

## 0. 结论先行

- **核心价值**：以不到 2% 的可训练参数量实现接近全量微调的效果（LoRA r=64，7B 模型仅需 ~12GB 显存训练），是在资源受限条件下对大模型进行领域适配的最优工程方案。
- **QLoRA 是消费级 GPU 的关键**：NF4 4-bit 量化底座 + LoRA，使在单张 48GB GPU 上微调 65B 参数模型成为可能，对 Tri-Transformer 阶段 2 联合微调至关重要。
- **Tri-Transformer 缝合训练选型**：阶段 1（C-Transformer 预热）不启用 LoRA；阶段 2（LoRA 联合微调）I/O 两端用 r=32, alpha=64 LoRA；阶段 3（可选端到端）解冻底层 1/4 后全量。A100 80G 可完整承载阶段 2（~61 GB）。
- **多模态对齐推荐**：跨模态分布差异大，rank 建议 32-64（高于普通指令微调的 8-16），alpha 设为 2× rank。

---

## 1. 概述

**LoRA（Low-Rank Adaptation）** 是 Microsoft 研究院于 2021 年提出的参数高效微调（PEFT）方法（arXiv:2106.09685，ICLR 2022），通过在 Transformer 层中注入**可训练的低秩矩阵**，以极小的参数量实现与全量微调相当的效果。

**QLoRA** 是在 LoRA 基础上引入 4-bit 量化的进一步优化（arXiv:2305.14314，NeurIPS 2023），使得在单张 48GB GPU 上微调 65B 参数模型成为可能。

**在 Tri-Transformer 中的角色**：C-Transformer 与 I/O 两端衔接层的"缝合训练"阶段——即将两端异构大模型权重与中间控制层进行联合对齐时，LoRA 是计算成本最低的渐进式微调方案。

---

## 2. 技术原理

### 2.1 LoRA 的低秩分解

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，冻结 $W_0$，注入可训练低秩矩阵：

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

其中：
- $A \in \mathbb{R}^{r \times k}$（输入投影，高斯初始化）
- $B \in \mathbb{R}^{d \times r}$（输出投影，**零初始化**，保证训练初始时 $\Delta W = 0$）
- $r \ll \min(d, k)$（秩，通常 4-64）
- $\alpha$：缩放因子（通常设为 $r$ 或 $2r$）

### 2.2 可训练参数量对比

| 方法 | 可训练参数 | 显存（FP16 训练） | 训练速度 |
|---|---|---|---|
| 全量微调（7B） | 7B（100%） | ~80 GB | 1× |
| LoRA（r=8，7B） | ~17M（0.24%） | ~24 GB | ~0.95× |
| LoRA（r=64，7B） | ~134M（1.9%） | ~28 GB | ~0.90× |
| QLoRA（4bit + r=64，7B） | ~134M（1.9%） | **~12 GB** | ~0.80× |
| QLoRA（4bit + r=64，70B） | ~1.3B（1.9%） | **~48 GB** | ~0.75× |

### 2.3 完整 PyTorch LoRA 实现

```python
import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias_param)
        if self.r > 0 and not self.merged:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_out + lora_out * self.scaling
        return base_out

    def merge_weights(self):
        if not self.merged and self.r > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def unmerge_weights(self):
        if self.merged and self.r > 0:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def extra_repr(self) -> str:
        return f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, r={self.r}"
```

### 2.4 LoRA rank 选择经验法则

| 任务类型 | 推荐 rank | 推荐 alpha | 说明 |
|---|---|---|---|
| 指令跟随微调 | 8-16 | 16-32 | 通用对话 |
| 垂直领域知识注入 | 16-32 | 32-64 | 医疗/法律/金融 |
| 代码生成 | 16-64 | 32-128 | 需要较大容量 |
| 风格/人格迁移 | 4-8 | 8-16 | 参数效率优先 |
| 多模态对齐（本项目）| 32-64 | 64-128 | 跨模态分布差异大 |
| 超长上下文适配 | 8-16 | 16-32 | YaRN/扩窗微调 |

**经验法则**：
- `alpha ≈ 2 × r` 是最常用设置
- rank 越大：拟合能力越强，过拟合风险越高，建议配合更大数据集
- 多模态任务因模态分布差异大，rank 建议设置偏高（32-64）

### 2.5 LoRA 应用于哪些层

```python
def apply_lora_to_transformer(
    model: nn.Module,
    r: int = 16,
    lora_alpha: float = 32.0,
    target_modules: list = None,
    lora_dropout: float = 0.05,
):
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        lora_layer = LoRALinear(
            module.in_features, module.out_features,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias=module.bias is not None,
        )
        lora_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_layer.bias_param.data = module.bias.data.clone()
        setattr(parent, attr_name, lora_layer)
        replaced += 1
    print(f"LoRA: replaced {replaced} linear layers (r={r}, alpha={lora_alpha})")
    return model
```

---

## 3. QLoRA：4-bit 量化 + LoRA

### 3.1 核心技术

**① NF4（NormalFloat4）**：
- 专为正态分布权重设计的信息论最优 4-bit 量化
- 将 FP16 权重量化到 16 个分位点，最大化信息保留
- 精度损失：NF4 ppl 增量 ≈ 0.3-0.8（vs FP16）

**② Double Quantization（双重量化）**：
- 对量化常数（Scale Factor）再次量化，平均节省 0.37 bits/参数

**③ Paged Optimizer**：
- 梯度检查点激活时的内存峰值通过 NVIDIA 统一内存自动 Offload 到 CPU

### 3.2 使用 peft + bitsandbytes 实现 QLoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 4. LoRA 变体对比

| 变体 | 核心改进 | 相比标准 LoRA | 适用场景 |
|---|---|---|---|
| **LoRA**（原版） | 低秩矩阵注入 | 基准 | 通用 PEFT |
| **LoRA+**（2024） | A/B 矩阵使用不同学习率（B 更大） | 收敛更快，效果略提升 | 数据充足场景 |
| **DoRA**（2024） | 将权重分解为幅度+方向，LoRA 仅调整方向 | 代码/数学任务提升明显 | 推理密集任务 |
| **MoLoRA**（2024） | 多个 LoRA 分支 + MoE 门控 | 多任务微调效果更好 | 多任务场景 |
| **LISA**（2024） | 随机层全量更新，其余冻结 | 效果优于 LoRA，实现更简单 | 资源受限场景 |
| **QLoRA** | 4-bit 量化底座 + LoRA | 显存减少 50-70% | 消费级 GPU |

### 4.1 DoRA 简化实现

```python
class DoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 16):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.magnitude = nn.Parameter(
            self.weight.norm(p=2, dim=0, keepdim=True).clone()
        )
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = 1.0 / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        W_adapted = self.weight + delta_W
        W_norm = W_adapted.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8)
        W_direction = W_adapted / W_norm
        W_final = self.magnitude * W_direction
        return nn.functional.linear(x, W_final)
```

---

## 5. Tri-Transformer 缝合训练方案

### 5.1 三阶段训练流程

```
┌─────────────────────────────────────────────────────┐
│ 阶段 1：C-Transformer 预热（约 10K steps）           │
│   冻结：I-Transformer（大模型A）全部参数             │
│          O-Transformer（大模型B）全部参数             │
│   训练：C-Transformer + 衔接层（从零初始化）         │
│   LoRA：不启用（C 本身参数量不大，直接训练）          │
│   学习率：3e-4                                        │
│   目标：C 学会从 I 读信号、向 O 发信号               │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 2：LoRA 联合微调（约 50K steps）                │
│   冻结：I-Transformer 全部参数                       │
│          O-Transformer 全部参数                       │
│   训练：C-Transformer + 衔接层                       │
│   LoRA：I 和 O 两端 Attention 层（r=32, alpha=64）  │
│   学习率：I/O LoRA: 1e-4, C: 3e-4                   │
│   目标：两端大模型在 C 的调制下协同工作              │
└─────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 3：全量端到端微调（可选，约 20K steps）          │
│   解冻：I/O 底层 1/4 层（Transformer 底层更通用）    │
│   训练：全模型端到端，学习率降低至 1e-5              │
│   LoRA：可合并后继续全量微调，或保持 LoRA 形式       │
│   目标：最大化端到端性能，消除模态接缝              │
└─────────────────────────────────────────────────────┘
```

### 5.2 各阶段配置建议表

| 参数 | 阶段 1 | 阶段 2 | 阶段 3 |
|---|---|---|---|
| 学习率（C-Transformer） | 3e-4 | 3e-4 | 1e-5 |
| 学习率（LoRA A/B） | — | 1e-4 | 1e-5 |
| 学习率（已解冻层） | — | — | 1e-5 |
| LoRA rank | — | 32 | 32（或 merge） |
| LoRA alpha | — | 64 | 64（或 merge） |
| 批量大小 | 32 | 16 | 8 |
| 梯度检查点 | 否 | 是 | 是 |
| warmup steps | 500 | 1000 | 500 |

### 5.3 保存与加载 LoRA 权重

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model.save_pretrained("./tri_transformer_lora_stage2")

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
peft_model = PeftModel.from_pretrained(base_model, "./tri_transformer_lora_stage2")

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./tri_transformer_merged")
```

### 5.4 显存估算（A100 80G，阶段 2）

| 组件 | 参数量 | 显存占用（BF16 + Adam） |
|---|---|---|
| I-Transformer（Qwen3-8B，冻结） | 8B | ~16 GB（仅推理） |
| O-Transformer（Qwen3-8B，冻结） | 8B | ~16 GB（仅推理） |
| C-Transformer（训练） | ~500M | ~6 GB |
| LoRA（I + O，r=32） | ~268M | ~3.2 GB |
| 激活/梯度（batch=16） | — | ~20 GB |
| **总计** | — | **~61 GB** |

> A100 80G 可承载完整阶段 2 训练（使用 gradient checkpointing 可降低激活占用）。

---

## 6. 最新进展（2024-2025）

### 6.1 LoRA+（2024）
- 将 LoRA 的 A 矩阵和 B 矩阵使用不同学习率（B 矩阵学习率 = A 矩阵的 2-16×），理论上收敛更快，微调质量更高。

### 6.2 DoRA（Weight-Decomposed Low-Rank Adaptation，2024）
- 将权重分解为幅度（Magnitude）和方向（Direction）两个分量，LoRA 仅调整方向，幅度可学习。在代码生成、数学推理等任务上优于标准 LoRA。

### 6.3 MoLoRA / MOELoRA（2024）
- 为每个训练任务准备多个 LoRA 分支，通过门控机制动态选择，类似 MoE 结构，适合多任务微调场景。

### 6.4 LISA（2024）
- 随机层选择策略：每步只激活少数几层进行全量更新，其余层冻结，达到比 LoRA 更好的效果，且实现更简单。

### 6.5 LoRA 在多模态的应用
- **LLaVA-1.5、Qwen-VL** 等视觉语言模型均使用 LoRA 对视觉投影层和语言模型进行联合微调，是 Tri-Transformer 多模态对齐训练的重要参考。
