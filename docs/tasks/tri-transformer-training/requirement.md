# 需求文档 - Tri-Transformer 模型训练流程实现

## 背景

Tri-Transformer 三分支模型（ITransformer / CTransformer / OTransformer）已有完整的 PyTorch 架构实现（`backend/app/model/`），但当前 `TriTransformerTrainer` 使用随机虚拟数据（`_make_dummy_batch`）进行训练，缺乏真实数据集支持。本需求旨在：

1. 接入 ModelScope 上的中文对话/问答数据集
2. 集成本地 ollama 部署的 LLM 用于数据增强
3. 将 TextTokenizer 升级为基于 Qwen2.5 BPE 的真实分词器
4. 提供独立可运行的训练入口脚本

## 核心功能

### F1: ModelScope 数据集加载器
- 支持 LCCC（大规模中文对话）、BELLE（中文指令数据）、DuReader（阅读理解）
- 提供 PyTorch DataLoader，输出 `(src, tgt_in, tgt_out)` tensor

### F2: Trainer 数据集集成
- 替换 `_make_dummy_batch` 为真实数据集 batch 构造
- 支持 `padding`、`truncation`、`max_len` 限制

### F3: Ollama LLM 集成
- 通过 `http://localhost:11434` 调用本地 ollama HTTP API
- 可选模型：`gemma3:4b`、`modelscope.cn/Qwen/Qwen2.5-3B-Instruct-GGUF:latest`
- 用途：数据增强（生成训练标签/摘要）

### F4: 依赖安装脚本
- `backend/scripts/install_deps.sh`
- 安装：`torch`, `modelscope`, `transformers`, `datasets`, `huggingface-hub`

### F5: 训练入口脚本
- `backend/scripts/train.py`
- 参数：`--dataset lccc|belle|dureader`、`--epochs`、`--batch-size`、`--device`、`--max-steps`

### F6: Tokenizer 升级
- `TextTokenizer` 使用 `transformers.AutoTokenizer` 加载 Qwen2.5 词表
- vocab_size: 151936（需同步更新 `TriTransformerConfig` 默认值）

## 验收标准

| ID | 标准 |
|----|------|
| AC1 | DataLoader 从 ModelScope 加载数据集，输出 tensor batch |
| AC2 | Trainer 使用真实数据完成 ≥1 epoch 训练 |
| AC3 | Ollama 客户端成功调用 gemma3:4b 生成文本 |
| AC4 | `python backend/scripts/train.py --dataset lccc --epochs 3` 可运行 |
| AC5 | TextTokenizer encode() 输出 token_ids 在 [0, 151935] |
| AC6 | 现有 pytest 全部通过 |
| AC7 | install_deps.sh 执行后 torch/modelscope/transformers 可 import |

## 环境约束

- ollama v0.15.5，已部署：`gemma3:4b`, `Qwen2.5-3B-Instruct-GGUF`, `Qwen3-VL-4B`
- Python 3.10.12，PyTorch 未安装
- 国内网络，优先 ModelScope CDN 下载
- 训练支持 CPU fallback（无 GPU 时自动降级）
- 不破坏已有 FastAPI train API 兼容性
