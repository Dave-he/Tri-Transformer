# Tri-Transformer 训练与测试指南

## 📋 目录

- [快速开始](#快速开始)
- [训练模型](#训练模型)
- [评估模型](#评估模型)
- [模型推理](#模型推理)
- [测试套件](#测试套件)
- [故障排查](#故障排查)

---

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
cd backend
pip install -r requirements.txt

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 最小化测试（CPU）

```bash
# 轻量级配置，5 个 epoch，快速验证
python -m app.services.model.quick_start \
    --config lightweight \
    --epochs 5 \
    --batch-size 8 \
    --num-samples 500
```

---

## 🏋️ 训练模型

### 训练配置选项

| 配置 | 说明 | 推荐值 |
|------|------|--------|
| `lightweight` | 轻量研究版（~5M 参数） | 开发测试 |
| `qwen3-8b` | Qwen3-8B 规格（~16GB 显存） | 生产环境 |
| `qwen3-30b` | Qwen3-30B MoE（~60GB 显存） | 大规模部署 |

### GPU 训练（推荐）

```bash
# 使用 GPU，混合精度训练
python -m app.services.model.quick_start \
    --config lightweight \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --use-amp \
    --gradient-accumulation 2 \
    --num-samples 5000
```

### 断点续训

```bash
# 从最新检查点恢复
python -m app.services.model.quick_start \
    --resume ./checkpoints/checkpoint_latest.pt
```

### 训练参数详解

```bash
python -m app.services.model.quick_start --help
```

**关键参数：**

- `--config`: 模型配置（`lightweight` / `qwen3-8b` / `qwen3-30b`）
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--seq-len`: 序列长度
- `--use-amp`: 启用混合精度训练（节省显存）
- `--gradient-accumulation`: 梯度累积步数（模拟更大 batch）
- `--num-samples`: 训练样本数
- `--val-split`: 验证集比例
- `--output-dir`: 检查点输出目录

### 训练输出

训练过程中会保存以下文件：

```
checkpoints/
├── checkpoint_epoch1.pt          # 第 1 轮检查点
├── checkpoint_epoch2.pt          # 第 2 轮检查点
├── checkpoint_best.pt            # 最佳模型
├── checkpoint_latest.pt          # 最新检查点
├── checkpoint_final.pt           # 最终检查点
└── training_history.json         # 训练历史
```

---

## 📊 评估模型

### 运行评估

```bash
python -m app.services.model.evaluate \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --num-samples 200 \
    --output ./evaluation_results.json
```

### 评估指标

| 指标 | 说明 | 优劣 |
|------|------|------|
| **Perplexity** | 困惑度，衡量预测不确定性 | ↓ 越低越好 |
| **Accuracy** | Token 预测准确率 | ↑ 越高越好 |
| **Hallucination Rate** | 幻觉率，衡量生成内容真实性 | ↓ 越低越好 |
| **Retrieval Precision** | 检索精度 | ↑ 越高越好 |
| **Retrieval Recall** | 检索召回率 | ↑ 越高越好 |

### 评估报告示例

```
======================================================================
                            📊 评估报告                            
======================================================================

SYNTHETIC 数据集:
----------------------------------------------------------------------
  Perplexity (困惑度):  45.23
  Accuracy (准确率):    0.7834
  Hallucination Rate:   0.1245

======================================================================

评估指标说明:
  - Perplexity: 越低越好，表示模型预测更准确
  - Accuracy: 越高越好，表示 token 预测准确率
  - Hallucination Rate: 越低越好，表示幻觉内容更少
======================================================================
```

---

## 💬 模型推理

### 交互式对话

```bash
python -m app.services.model.inference_cli \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --mode interactive
```

**示例对话：**

```
============================================================
                   Tri-Transformer 交互式对话                   
============================================================
输入 'quit' 或 'exit' 退出
输入 'clear' 清空对话历史
============================================================

👤 您：你好，请介绍一下自己
🤖 AI: 你好！我是 Tri-Transformer，一个基于三分支 Transformer 架构的 AI 模型...

👤 您：什么是人工智能？
🤖 AI: 人工智能（AI）是计算机科学的一个分支，旨在创建能够执行需要人类智能任务的系统...
```

### 单句推理

```bash
python -m app.services.model.inference_cli \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --mode single \
    --input "如何学习机器学习？" \
    --temperature 0.7 \
    --max-length 128
```

### 批量推理

```bash
python -m app.services.model.inference_cli \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --mode batch
```

### 推理参数

- `--temperature`: 温度参数（>1 增加随机性，<1 减少随机性）
- `--top-k`: Top-k 采样（默认 50）
- `--top-p`: Nucleus 采样（默认 0.95）
- `--max-length`: 最大生成长度
- `--num-sequences`: 生成序列数

---

## 🧪 测试套件

### 运行完整测试

```bash
cd backend
pytest tests/test_complete_training.py -v -s
```

### 测试覆盖

测试套件包括：

1. **模型测试** (`TestTriTransformerModel`)
   - 前向传播
   - 输出形状验证
   - 掩码处理
   - 参数量计算

2. **损失函数测试** (`TestLossFunctions`)
   - Hallucination Loss
   - RAG Loss
   - Control Alignment Loss
   - Total Loss

3. **评估器测试** (`TestEvaluators`)
   - Perplexity Evaluator
   - Accuracy Evaluator
   - Hallucination Evaluator
   - RAG Evaluator

4. **训练器测试** (`TestAdvancedTrainer`)
   - 训练器初始化
   - 单轮训练
   - 评估流程
   - 检查点保存/加载
   - 完整训练循环

5. **集成测试** (`TestIntegration`)
   - 端到端训练流程

### 运行特定测试

```bash
# 只测试模型前向传播
pytest tests/test_complete_training.py::TestTriTransformerModel -v

# 只测试损失函数
pytest tests/test_complete_training.py::TestLossFunctions -v

# 只测试训练器
pytest tests/test_complete_training.py::TestAdvancedTrainer -v
```

---

## 🔧 故障排查

### 常见问题

#### 1. CUDA Out of Memory

**解决方案：**

```bash
# 减小 batch size
--batch-size 4

# 启用混合精度训练
--use-amp

# 使用梯度累积
--gradient-accumulation 4
```

#### 2. 训练 Loss 不下降

**可能原因：**
- 学习率过高/过低
- 数据质量问题
- 模型容量不足

**解决方案：**
```bash
# 调整学习率
--lr 5e-5

# 增加训练样本
--num-samples 10000
```

#### 3. 评估 Perplexity 过高

**可能原因：**
- 训练不足
- 过拟合训练数据

**解决方案：**
```bash
# 增加训练轮数
--epochs 30

# 添加早停
# （在代码中设置 early_stopping_patience=5）
```

#### 4. 检查点加载失败

**解决方案：**
```bash
# 确保使用相同的配置
python -m app.services.model.quick_start \
    --config lightweight \
    --resume ./checkpoints/checkpoint_best.pt
```

---

## 📈 性能优化建议

### GPU 训练优化

1. **启用混合精度**
   ```bash
   --use-amp  # 节省 50% 显存，提升 2-3x 速度
   ```

2. **梯度累积**
   ```bash
   --gradient-accumulation 4  # 模拟 4x batch size
   ```

3. **多 GPU 训练**（未来支持）
   ```bash
   # TODO: 支持 DDP 分布式训练
   ```

### 推理优化

1. **KV Cache**（已支持）
   - 模型内置 KV Cache 机制
   - 流式推理速度提升 5-10x

2. **量化**（未来支持）
   ```bash
   # TODO: 支持 INT8/FP16 量化
   ```

---

## 📚 进阶使用

### 自定义数据集训练

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        # 加载数据
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回 (src, tgt_in, tgt_out)
        return src_ids, tgt_in_ids, tgt_out_ids

# 创建数据加载器
dataset = CustomDataset("data.json", tokenizer, max_len=128)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 训练
trainer.train(train_loader=loader, num_epochs=10)
```

### 自定义损失函数

```python
from app.services.model.loss_functions import TotalLoss

# 调整损失权重
custom_loss = TotalLoss(
    llm_weight=1.0,
    hallucination_weight=0.8,  # 增加幻觉检测权重
    rag_weight=0.5,            # 增加 RAG 权重
    control_weight=0.3,
)
```

---

## 📞 支持与反馈

如有问题，请提交 Issue 或联系开发团队。

**相关文档：**
- [架构文档](../../docs/agent/architecture.md)
- [开发指南](../../docs/agent/development_commands.md)
- [测试规范](../../docs/agent/testing.md)
