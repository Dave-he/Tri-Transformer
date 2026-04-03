# DeepSpeed ZeRO-3（零冗余优化器）

## 0. 结论先行

- **核心作用**：将优化器状态、梯度、模型参数三类训练状态完全分片到所有 GPU，理论最高降低单卡显存 64×，是在有限 GPU 资源上训练超大规模模型的核心基础设施。
- **选型建议**：ZeRO-1（仅优化器分片）→ ZeRO-2（+ 梯度分片）→ ZeRO-3（+ 参数分片），显存限制越紧选阶段越高；ZeRO-3 + CPU Offload 是单机多卡训练超大模型的最佳方案，代价是约 30-40% 通信开销。
- **FSDP 对比**：DeepSpeed ZeRO-3 功能更完整（CPU/NVMe Offload）；PyTorch FSDP 原生集成更便利，适合中型模型（≤ 30B）；两者性能相近，选 FSDP 更易维护，选 DeepSpeed 功能更全。
- **Tri-Transformer 中的角色**：I/C/O-Transformer 三分支 + 两端大模型联合训练时的分布式内存管理基础设施；LoRA 缝合训练阶段（`19_lora_qlora.md`）的配套工具，Phase 1 单机可用 ZeRO-2，联合全量训练需 ZeRO-3。

---

## 1. 概述

DeepSpeed ZeRO（Zero Redundancy Optimizer）是 Microsoft 开发的大模型分布式训练内存优化技术（arXiv:1910.02054，SC 2020 最佳论文）。ZeRO-3 是其最高阶段，通过将**优化器状态、梯度、模型参数**三类训练状态完全分片到所有 GPU，理论上可将单卡显存需求降低最高 64 倍，使得在有限 GPU 资源上训练超大规模模型成为可能。

**在 Tri-Transformer 中的角色**：Tri-Transformer 三分支架构（I/C/O-Transformer + 两端插拔大模型）参数量极大，ZeRO-3 是实现多节点分布式联合训练的核心基础设施。

---

## 2. 核心原理

### 2.1 数据并行的内存冗余

标准数据并行（DDP）中，每张 GPU 保存一份完整的模型副本：

```
GPU 0: [模型参数 W] [梯度 ∇W] [优化器状态 m,v（Adam）]
GPU 1: [模型参数 W] [梯度 ∇W] [优化器状态 m,v（Adam）]
GPU 2: [模型参数 W] [梯度 ∇W] [优化器状态 m,v（Adam）]
GPU 3: [模型参数 W] [梯度 ∇W] [优化器状态 m,v（Adam）]
```

对于 FP16 训练，7B 参数模型需要约：
- 参数：14 GB（FP16）
- 梯度：14 GB（FP16）
- 优化器状态：56 GB（Adam FP32：参数 + 一阶矩 + 二阶矩）
- **总计：84 GB/GPU（×4 GPU = 336 GB 纯冗余）**

### 2.2 ZeRO 三阶段分片

```
Stage 1（优化器分片）：
GPU 0: [W] [∇W] [OS 0/4]   ← 仅保存 1/4 的优化器状态
GPU 1: [W] [∇W] [OS 1/4]
...                          → 内存节省 ~4x

Stage 2（梯度 + 优化器分片）：
GPU 0: [W] [∇W 0/4] [OS 0/4]
GPU 1: [W] [∇W 1/4] [OS 1/4]
...                          → 内存节省 ~8x

Stage 3（参数 + 梯度 + 优化器分片）：
GPU 0: [W 0/4] [∇W 0/4] [OS 0/4]
GPU 1: [W 1/4] [∇W 1/4] [OS 1/4]
GPU 2: [W 2/4] [∇W 2/4] [OS 2/4]
GPU 3: [W 3/4] [∇W 3/4] [OS 3/4]
                             → 内存节省 ~64x
```

### 2.3 ZeRO-3 前向/反向传播通信

ZeRO-3 的参数分片要求在前向和反向传播时动态收集（All-Gather）当前层所需参数：

```
前向传播（每层）:
All-Gather(W_i) → 计算激活值 → 释放 W_i（仅保留本 GPU 分片）

反向传播（每层）:
All-Gather(W_i) → 计算梯度 ∇W_i → Reduce-Scatter(∇W_i) → 释放
```

**通信量分析**：ZeRO-3 的通信量约为标准 DDP 的 1.5 倍，但显存节省通常超过通信额外开销。

---

## 3. 配置方法

### 3.1 DeepSpeed 配置文件（ds_config.json）

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 128,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true,

    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },

    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 3.2 训练代码集成

```python
import deepspeed
import torch
import torch.nn as nn

class TriTransformerTrainer:
    def __init__(self, model, ds_config_path: str):
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config_path
        )

    def train_step(self, batch):
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        outputs = self.engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        self.engine.backward(loss)
        self.engine.step()
        return loss.item()

    def save_checkpoint(self, save_dir: str, tag: str = "best"):
        self.engine.save_checkpoint(save_dir, tag=tag)

    @staticmethod
    def load_checkpoint(model, save_dir: str, tag: str = "best"):
        _, client_sd = model.load_checkpoint(save_dir, tag=tag)
        return client_sd
```

### 3.3 启动脚本

```bash
deepspeed --num_gpus=8 \
          --num_nodes=4 \
          --master_addr $MASTER_ADDR \
          --master_port 29500 \
          train_tri_transformer.py \
          --deepspeed ds_config_zero3.json \
          --model_config configs/tri_transformer_7b.yaml
```

### 3.4 ZeRO-Infinity（NVMe Offload）

对于超超大模型（> 100B 参数），ZeRO-Infinity 支持将参数 Offload 到 NVMe SSD：

```json
"offload_param": {
    "device": "nvme",
    "nvme_path": "/nvme/zero_offload",
    "buffer_count": 5,
    "buffer_size": 1e8
}
```

---

## 4. 显存估算

对于 Tri-Transformer 目标规模（约 13B 参数，含两端插拔 7B 模型 × 2 + C-Transformer 1B）：

| 方案 | 每卡显存需求 | 最低 GPU 配置 |
|---|---|---|
| 标准 DDP（FP16） | ~312 GB | 不可行（无 GPU 支持） |
| ZeRO-2（8卡 A100 80G） | ~78 GB/卡 | 需要 8×H100 |
| ZeRO-3（8卡 A100 80G） | ~20 GB/卡 | **8×A100 40G 可行** |
| ZeRO-3 + CPU Offload（4卡） | ~12 GB/卡 | **4×RTX 4090 可行** |

---

## 5. 最新进展（2024-2025）

### 5.1 DeepSpeed-FastGen（2023）
- 面向推理的 DeepSpeed 分布式推理引擎，支持分布式 KV Cache，可在多节点上服务超大规模模型。

### 5.2 DeepSpeed-MoE（2023）
- 专为混合专家（MoE）架构优化的 ZeRO 实现，与 Mixtral 等 MoE 模型无缝集成。

### 5.3 与 FSDP（PyTorch 官方）的对比
- **PyTorch FSDP**（Fully Sharded Data Parallel）是 PyTorch 2.0 内置的类 ZeRO-3 实现，无需安装 DeepSpeed。
- FSDP 对 PyTorch 生态兼容性更好，DeepSpeed ZeRO-3 功能更丰富（CPU/NVMe offload、ZeRO-Infinity）。
- Tri-Transformer 推荐在 PyTorch 2.0+ 上同时评估两者。

### 5.4 MegaScale（ByteDance, 2024）
- 字节跳动发布的万卡大规模训练基础设施论文，在 ZeRO-3 基础上针对网络故障恢复、通信优化等工程问题进行了深度定制，是工业级参考。
