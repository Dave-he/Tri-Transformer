# 任务清单 - api-fix-and-supplement-v1

## GREEN 阶段（后端修复）

| ID | 任务 | 文件 |
|----|------|------|
| T1 | [GREEN] 实现 DoraAdapter 类 | app/model/lora_adapter.py |
| T2 | [GREEN] 实现 validate_galore_config 函数 | app/services/train/train_service.py |
| T3 | [GREEN] 实现 HippoRetriever 类 | app/services/rag/retriever.py |
| T4 | [GREEN] 新增 GET /chat/sessions 端点 | chat.py + chat_service.py + schemas/chat.py |

## VERIFY 阶段

| ID | 任务 |
|----|------|
| T5 | 后端全量 pytest 0 failures |
| T6 | 前端全量 vitest 115 passed |
| T7 | flake8 增量 0 errors |
