# 任务清单 - tri-transformer-feature-v1

## RED 阶段

| ID | 任务 | 文件 |
|----|------|------|
| T1 | [RED] GET /metrics 正常返回 | tests/test_metrics.py |
| T2 | [RED] GET /training/status 正常返回 | tests/test_metrics.py |
| T3 | [RED] 无认证 401 | tests/test_metrics.py |
| T4 | [RED] POST /jobs/start 返回 jobId | tests/test_train.py |
| T5 | [RED] GET /jobs/progress 返回进度 | tests/test_train.py |
| T6 | [RED] GET /jobs/models 返回模型列表 | tests/test_train.py |

## GREEN 阶段（后端）

| ID | 任务 | 文件 |
|----|------|------|
| T7 | [GREEN] 新建 metrics.py | app/api/v1/metrics.py |
| T8 | [GREEN] train.py 新增 3 路由 | app/api/v1/train.py |
| T9 | [GREEN] main.py 注册 metrics router | app/main.py |

## GREEN 阶段（前端）

| ID | 任务 | 文件 |
|----|------|------|
| T10 | [GREEN] conversations.ts 字段映射 | src/api/conversations.ts |
| T11 | [GREEN] types/api.ts 新增字段 | src/types/api.ts |
| T12 | [GREEN] MessageBubble 幻觉 Tag | src/components/chat/MessageBubble.tsx |
| T13 | [GREEN] trainingConfig.ts 路径对齐 | src/api/trainingConfig.ts + mocks |

## VERIFY 阶段

| ID | 任务 |
|----|------|
| T14 | 后端全量 pytest ≥ 191 passed |
| T15 | 前端全量 vitest 115 passed |
| T16 | flake8 增量 0 errors |
