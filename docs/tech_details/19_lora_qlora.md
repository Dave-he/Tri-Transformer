# LoRA & QLoRA（低秩适配微调）

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

对于 GPT-3 175B（$d=12288$，Transformer 层权重约为 $12288 \times 12288$）：

| 方法 | 可训练参数 | 内存（相对）| 速度（相对） |
|---|---|---|---|
| 全量微调 | 175B | 1× | 1× |
| LoRA（r=8） | ~17.5M（0.01%）| 1/3× | ~1× |
| LoRA（r=64） | ~140M（0.08%）| 1/2× | ~1× |
| QLoRA（4bit + r=64） | ~140M | **1/10×** | ~0.8× |

### 2.3 PyTorch 实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, lora_alpha: float = 16,
                 lora_dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_out + lora_out * self.scaling
        return base_out

    def merge_weights(self):
        """推理前合并 LoRA 权重到基础权重（消除推理额外开销）"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        del self.lora_A, self.lora_B
        self.r = 0
```

### 2.4 LoRA 应用于哪些层

```python
def apply_lora_to_transformer(model: nn.Module, r: int = 16,
                               lora_alpha: float = 32,
                               target_modules: list = None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent = get_parent_module(model, name)
                attr_name = name.split(".")[-1]
                lora_layer = LoRALinear(
                    module.in_features, module.out_features,
                    r=r, lora_alpha=lora_alpha
                )
                lora_layer.weight.data = module.weight.data.clone()
                setattr(parent, attr_name, lora_layer)
    return model
```

---

## 3. QLoRA：4-bit 量化 + LoRA

### 3.1 核心技术

**① NF4（NormalFloat4）数据类型**：
- 专为正态分布权重设计的信息论最优 4-bit 量化。
- 将 FP16 权重量化到 16 个分位点，最大化信息保留。

**② Double Quantization（双重量化）**：
- 对量化常数（Scale Factor）再次量化，平均节省 0.37 bits/参数。

**③ Paged Optimizer**：
- 梯度检查点激活时的内存峰值通过 NVIDIA 统一内存自动 Offload 到 CPU。

### 3.2 使用 peft + bitsandbytes 实现 QLoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## 4. Tri-Transformer 缝合训练方案

### 4.1 分阶段 LoRA 策略

```
阶段 1（对齐预热）:
    冻结: I-Transformer（大模型A）全部参数
          O-Transformer（大模型B）全部参数
    训练: C-Transformer + 衔接层（从零初始化）
    LoRA: 不启用（C 本身参数量不大）
    目标: C-Transformer 学会从 I 读信号、向 O 发信号

阶段 2（LoRA 联合微调）:
    冻结: I-Transformer 全部参数
          O-Transformer 全部参数
    训练: C-Transformer + 衔接层 + LoRA（A/B 两端 Attention 层）
    LoRA rank: r=16（衔接层精细调整）
    目标: 两端大模型在 C 的调制下协同工作

阶段 3（全量微调，可选）:
    解冻: I-Transformer 底层 + O-Transformer 底层
    训练: 全模型端到端（学习率降低 10×）
    目标: 最大化端到端性能
```

### 4.2 保存与加载 LoRA 权重

```python
from peft import PeftModel

model.save_pretrained("./tri_transformer_lora_stage2")

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
peft_model = PeftModel.from_pretrained(base_model, "./tri_transformer_lora_stage2")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./tri_transformer_merged")
```

---

## 5. 最新进展（2024-2025）

### 5.1 LoRA+ （2024）
- 将 LoRA 的 A 矩阵和 B 矩阵使用不同学习率（B 矩阵学习率 = A 矩阵的 2-16×），理论上收敛更快，微调质量更高。

### 5.2 DoRA（Weight-Decomposed Low-Rank Adaptation，2024）
- 将权重分解为幅度（Magnitude）和方向（Direction）两个分量，LoRA 仅调整方向，幅度可学习。在代码生成、数学推理等任务上优于标准 LoRA。

### 5.3 MoLoRA / MOELoRA（2024）
- 为每个训练任务准备多个 LoRA 分支，通过门控机制动态选择，类似 MoE 结构，适合多任务微调场景。

### 5.4 LISA（2024）
- 随机层选择策略：每步只激活少数几层进行全量更新，其余层冻结，达到比 LoRA 更好的效果，且实现更简单。
