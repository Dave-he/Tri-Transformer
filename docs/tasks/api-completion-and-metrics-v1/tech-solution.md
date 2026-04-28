# 技术方案 - api-completion-and-metrics-v1

## 设计决策

| ID | 决策 | 原因 |
|----|------|------|
| DD1 | Model API三端点采用轻量实现 | mock_inference=True下, 状态端点反映当前配置即可 |
| DD2 | Chat DELETE级联删除消息 | 避免孤儿数据 |
| DD3 | 后端路由/响应格式匹配前端期望 | 前端已有UI逻辑, 改后端更安全 |
| DD4 | BGEReranker通过settings.use_reranker开关 | mock模式下避免随机分数影响测试 |
| DD5 | SSE用async generator + StreamingResponse | FastAPI原生支持, 无需额外依赖 |
| DD6 | Train configs返回3种预设(default/lora/deepspeed) | 覆盖PRD常见场景 |
| DD7 | 删除metrics.ts, 统一training.ts | 两文件完全重复 |

## 文件变更概览

### 后端新增文件
- `backend/app/schemas/model.py` - ModelStatus/Info/Load schemas
- `backend/app/schemas/knowledge.py` - DocumentStatus/SearchRequest schemas
- `backend/app/schemas/train.py` - TrainConfigPreset schema
- `backend/app/services/model/model_service.py` - ModelService

### 后端修改文件
- `backend/app/api/v1/model.py` - +3端点
- `backend/app/api/v1/chat.py` - +DELETE端点, 修改send_message响应
- `backend/app/api/v1/knowledge.py` - +status端点, search改为POST, 修改响应格式
- `backend/app/api/v1/stream.py` - +SSE端点
- `backend/app/api/v1/train.py` - +configs端点
- `backend/app/services/chat/chat_service.py` - +delete_session, rerank集成
- `backend/app/services/rag/retriever.py` - +rerank步骤

### 前端新增文件
- `frontend/src/api/model.ts` - Model API client

### 前端修改文件
- `frontend/src/api/documents.ts` - 修正路由/字段映射
- `frontend/src/api/conversations.ts` - +deleteSessionApi, 修正响应格式
- `frontend/src/api/training.ts` - +model/config API函数
- `frontend/src/types/index.ts` - 新增类型定义

### 前端删除文件
- `frontend/src/api/metrics.ts` - 重复模块删除
