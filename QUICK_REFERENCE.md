# Tri-Transformer 快速参考

## 🚀 一行命令

### 验证模型
```bash
cd backend && python verify_model.py
```

### 快速训练（CPU）
```bash
python -m app.services.model.quick_start --config lightweight --epochs 5 --num-samples 500
```

### 快速训练（GPU）
```bash
python -m app.services.model.quick_start --config lightweight --epochs 10 --batch-size 16 --use-amp --num-samples 5000
```

### 评估模型
```bash
python -m app.services.model.evaluate --checkpoint ./checkpoints/checkpoint_best.pt
```

### 交互式推理
```bash
python -m app.services.model.inference_cli --checkpoint ./checkpoints/checkpoint_best.pt --mode interactive
```

### 运行测试
```bash
pytest tests/test_complete_training.py -v
```

---

## 📦 模型配置

| 配置 | 参数量 | 显存 | 用途 |
|------|--------|------|------|
| `lightweight` | ~14M | ~100MB | 开发测试 |
| `qwen3-8b` | ~16B | ~16GB | 生产环境 |
| `qwen3-30b` | 30B/3B | ~60GB | 大规模部署 |

---

## 🔧 关键参数

### 训练参数
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率（默认 1e-4）
- `--seq-len`: 序列长度（默认 64）
- `--num-samples`: 样本数量
- `--use-amp`: 混合精度训练
- `--gradient-accumulation`: 梯度累积

### 推理参数
- `--temperature`: 温度（0.7 推荐）
- `--top-k`: Top-k 采样（50）
- `--top-p`: Nucleus 采样（0.95）
- `--max-length`: 最大长度（128）

---

## 📊 评估指标

| 指标 | 含义 | 优劣 | 典型值 |
|------|------|------|--------|
| Perplexity | 困惑度 | ↓ 越低越好 | <100 |
| Accuracy | 准确率 | ↑ 越高越好 | >0.8 |
| Hallucination Rate | 幻觉率 | ↓ 越低越好 | <0.1 |

---

## 📁 输出文件

```
checkpoints/
├── checkpoint_epoch1.pt      # 第 1 轮检查点
├── checkpoint_best.pt        # 最佳模型
├── checkpoint_latest.pt      # 最新检查点
├── checkpoint_final.pt       # 最终检查点
└── training_history.json     # 训练历史
```

---

## 🐛 故障排查

### CUDA Out of Memory
```bash
--batch-size 4 --use-amp --gradient-accumulation 4
```

### Loss 不下降
```bash
--lr 5e-5 --epochs 30
```

### Perplexity 过高
```bash
--epochs 50 --num-samples 10000
```

---

## 📚 完整文档

- [训练指南](docs/agent/training_guide.md) - 详细教程
- [架构文档](docs/agent/architecture.md) - 技术细节
- [实现总结](MODEL_IMPLEMENTATION_SUMMARY.md) - 完成清单

---

**快速启动**: `python verify_model.py` → `quick_start` → `evaluate` → `inference_cli`
