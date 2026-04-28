# 任务清单 - api-completion-and-metrics-v1

## GREEN 阶段

| ID | 任务 | 核心文件 |
|----|------|---------|
| T1 | Model API三端点 + Service + Schema | model.py(schemas/service/api) |
| T2 | Chat DELETE端点 + 级联删除 + 响应格式修复 | chat.py(api/service/schemas) |
| T3 | Knowledge文档状态 + 搜索POST + 响应格式修复 | knowledge.py(schemas/api) |
| T4 | BGEReranker接入搜索管道 | retriever.py + chat_service.py + config.py |
| T5 | SSE流式输出端点 | stream.py |
| T6 | Train configs预设端点 | train.py(schemas/api) |
| T7 | 前端API路由对齐 + 类型修复 | documents.ts + conversations.ts + model.ts + types |
| T8 | 前端重复模块清理 | 删除metrics.ts |

## VERIFY 阶段

| ID | 任务 |
|----|------|
| T9 | 后端全量 pytest 0 failures |
| T10 | 前端全量 vitest 0 failures |
| T11 | flake8 增量 0 errors |
