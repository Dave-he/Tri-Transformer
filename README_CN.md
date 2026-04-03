# Tri-Transformer AI 幻觉检测与实时通信系统

基于三分支 Transformer 架构的 AI 幻觉检测与实时通信系统，提供高精度幻觉检测、RAG 知识库问答、WebSocket 实时通信，以及 React 数据可视化前端。

## 🎯 核心特性

### 三分支 Transformer 架构
- **ITransformer**: 输入编码器，流式因果编码
- **CTransformer**: 控制中枢，DiT 风格 State Slots
- **OTransformer**: 输出解码器，Planning + Streaming

### Qwen3 兼容架构
- RoPE 位置编码（θ=1,000,000）
- GQA 分组查询注意力
- QK-Norm 每头归一化
- SwiGLU FFN
- Pre-RMSNorm
- adaLN-Zero 无侵入调制

### 训练优化
- 三阶段训练策略
- 混合精度训练（AMP）
- 梯度累积
- 动态学习率调度
- 早停机制

### 评估体系
- Perplexity（困惑度）
- Accuracy（准确率）
- Hallucination Rate（幻觉率）
- RAG Retrieval Metrics（检索指标）
- Dialog Quality（对话质量）

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
cd backend
pip install -r requirements.txt

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. 验证模型

```bash
cd backend
python verify_model.py
```

**预期输出：**
```
✅ 所有验证通过！Tri-Transformer 模型工作正常
```

### 3. 快速训练

**CPU 测试：**
```bash
python -m app.services.model.quick_start \
    --config lightweight \
    --epochs 5 \
    --batch-size 8 \
    --num-samples 500
```

**GPU 训练（推荐）：**
```bash
python -m app.services.model.quick_start \
    --config lightweight \
    --epochs 20 \
    --batch-size 16 \
    --use-amp \
    --num-samples 5000
```

### 4. 评估模型

```bash
python -m app.services.model.evaluate \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --num-samples 200
```

### 5. 交互式推理

```bash
python -m app.services.model.inference_cli \
    --checkpoint ./checkpoints/checkpoint_best.pt \
    --mode interactive
```

---

## 📦 模型规格

| 配置 | 参数量 | 显存 | 用途 |
|------|--------|------|------|
| **lightweight** | ~14M | ~100MB | 开发测试 |
| **qwen3-8b** | ~16B | ~16GB | 生产环境 |
| **qwen3-30b** | 30B/3B | ~60GB | 大规模部署 |

---

## 📚 文档导航

### 快速参考
- [快速参考卡片](QUICK_REFERENCE.md) - 一行命令速查
- [实现总结](MODEL_IMPLEMENTATION_SUMMARY.md) - 完成清单

### 详细文档
- [训练指南](docs/agent/training_guide.md) - 完整训练教程
- [架构文档](docs/agent/architecture.md) - 技术细节
- [开发指南](docs/agent/development_commands.md) - 开发命令
- [测试规范](docs/agent/testing.md) - 测试标准

---

## 🧪 测试

### 运行完整测试套件
```bash
cd backend
pytest tests/test_complete_training.py -v
```

### 运行特定测试
```bash
# 模型测试
pytest tests/test_complete_training.py::TestTriTransformerModel -v

# 损失函数测试
pytest tests/test_complete_training.py::TestLossFunctions -v

# 训练器测试
pytest tests/test_complete_training.py::TestAdvancedTrainer -v
```

---

## 📊 训练输出

训练完成后会生成以下文件：

```
checkpoints/
├── checkpoint_epoch1.pt      # 第 1 轮检查点
├── checkpoint_epoch2.pt      # 第 2 轮检查点
├── checkpoint_best.pt        # 最佳模型
├── checkpoint_latest.pt      # 最新检查点
├── checkpoint_final.pt       # 最终检查点
└── training_history.json     # 训练历史
```

---

## 🔧 故障排查

### CUDA Out of Memory
```bash
# 减小 batch size，启用混合精度，使用梯度累积
python -m app.services.model.quick_start \
    --batch-size 4 \
    --use-amp \
    --gradient-accumulation 4
```

### 训练 Loss 不下降
```bash
# 调整学习率，增加训练轮数
python -m app.services.model.quick_start \
    --lr 5e-5 \
    --epochs 30
```

### 评估 Perplexity 过高
```bash
# 增加训练样本和轮数
python -m app.services.model.quick_start \
    --num-samples 10000 \
    --epochs 50
```

---

## 📁 项目结构

```
Tri-Transformer/
├── backend/                      # FastAPI 后端
│   ├── app/
│   │   ├── model/               # 模型定义
│   │   │   ├── tri_transformer.py
│   │   │   ├── branches.py
│   │   │   └── trainer.py
│   │   └── services/model/      # 训练/评估/推理
│   │       ├── loss_functions.py
│   │       ├── evaluation.py
│   │       ├── train.py
│   │       ├── quick_start.py
│   │       ├── evaluate.py
│   │       └── inference_cli.py
│   └── tests/                   # 测试套件
│       └── test_complete_training.py
├── frontend/                     # React 前端
├── docs/agent/                   # 文档
│   ├── training_guide.md
│   ├── architecture.md
│   └── ...
├── MODEL_IMPLEMENTATION_SUMMARY.md
├── QUICK_REFERENCE.md
└── README.md
```

---

## 🎓 技术亮点

### 1. 三分支扭合架构
- I-Transformer 编码输入
- C-Transformer 生成控制信号
- O-Transformer 受控生成
- adaLN-Zero 实现无侵入调制

### 2. 幻觉检测机制
- 对比学习损失函数
- 事实一致性评估
- 知识引用检测
- 矛盾识别

### 3. RAG 集成
- 检索器 - 生成器联合训练
- 检索质量评估
- 知识 grounding

### 4. 控制对齐
- Thinking Mode 动态切换
- Response Style 调节
- Knowledge Grounding 控制

---

## 🔮 后续优化方向

### 短期（1-2 周）
- [ ] 真实数据集集成（LCCC、BelleGroup）
- [ ] Tokenizer 集成（SentencePiece/BPE）
- [ ] 多 GPU 训练（DDP）
- [ ] TensorBoard 可视化

### 中期（1 个月）
- [ ] LoRA 微调
- [ ] 模型量化（INT8/FP16）
- [ ] 推理优化
- [ ] RAG 集成（Milvus/ChromaDB）

### 长期（2-3 个月）
- [ ] Thinking Mode 实现
- [ ] 流式输出优化
- [ ] 生产环境部署
- [ ] API 服务优化

---

## 📞 支持与反馈

如有问题或建议：
1. 查看 [训练指南](docs/agent/training_guide.md)
2. 运行 `python verify_model.py` 验证环境
3. 提交 Issue 或联系开发团队

---

## 📄 许可证

本项目采用 [许可证名称] 许可证。

---

**状态**: ✅ 模型搭建完成，训练与测试流程已验证  
**最后更新**: 2026-04-03
