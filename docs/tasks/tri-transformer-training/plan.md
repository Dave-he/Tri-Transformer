# 任务清单 - Tri-Transformer 模型训练流程实现

## 概述

| 统计 | 数量 |
|------|------|
| P0 测试任务 | 5 |
| P0 代码任务 | 7 |
| P1 任务 | 1 |
| 总计 | 13 |

**测试命令**: `cd backend && python -m pytest tests/test_tri_transformer_forward.py tests/test_text_tokenizer.py tests/test_ollama_client.py tests/test_dataset_loader.py tests/test_trainer_with_dataloader.py -v`

---

## TDD 执行顺序

### 第一轮：RED（先写所有测试文件）

| ID | 类型 | 文件 | 说明 |
|----|------|------|------|
| T1-1 | test | `backend/tests/test_tri_transformer_forward.py` | forward() tuple 解包修复验证 |
| T2-1 | test | `backend/tests/test_text_tokenizer.py` | BPE tokenizer 升级验证 |
| T3-1 | test | `backend/tests/test_ollama_client.py` | ollama HTTP 客户端验证 |
| T4-1 | test | `backend/tests/test_dataset_loader.py` | ModelScope 数据加载验证 |
| T5-1 | test | `backend/tests/test_trainer_with_dataloader.py` | Trainer+DataLoader 集成验证 |

### 第二轮：GREEN（实现代码，让测试通过）

| ID | 类型 | 文件 | 依赖 |
|----|------|------|------|
| P0-1 | code | `backend/app/model/tri_transformer.py` | T1-1 |
| P0-2 | code | `backend/app/model/tokenizer/text_tokenizer.py` | T2-1 |
| P0-3 | code | `backend/app/services/model/ollama_client.py` | T3-1 |
| P0-4 | code | `backend/app/services/train/dataset_loader.py` | T4-1, P0-2 |
| P0-5 | code | `backend/app/model/trainer.py` | T5-1, P0-4 |
| P0-6 | code | `backend/scripts/install_deps.sh` | - |
| P0-7 | code | `backend/scripts/train.py` | P0-4, P0-5 |

### 第三轮：REFACTOR（修复受影响的测试）

| ID | 类型 | 文件 | 说明 |
|----|------|------|------|
| P1-1 | code | `backend/tests/test_tokenizer.py` | 更新 TEXT_MAX = 151936 |

---

## 验收标准

- AC1: DataLoader 输出 (src, tgt_in, tgt_out) torch.long tensor
- AC2: Trainer 使用真实数据完成 1 epoch，loss 有限
- AC3: OllamaClient.generate 返回非空字符串
- AC4: `python backend/scripts/train.py --dataset dummy --epochs 1` 可运行
- AC5: TextTokenizer encode() ids 在 [0, 151935]
- AC6: 现有 pytest 全部通过
- AC7: install_deps.sh 执行后可 import torch
