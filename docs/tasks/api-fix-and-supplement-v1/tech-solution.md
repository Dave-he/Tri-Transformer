# 技术方案：修复失败测试 + 补全缺失API端点

## 设计决策

| ID | 决策 | 原因 |
|----|------|------|
| DD1 | 实现DoraAdapter类(继承LoraAdapter架构+magnitude) | 测试明确要求DoRA论文设计 |
| DD2 | 在train_service.py新增validate_galore_config | 测试明确import路径 |
| DD3 | 在retriever.py新增HippoRetriever(PPR重排序) | HippoRAG核心算法 |
| DD4 | 在chat.py新增GET /sessions + ChatService.list_sessions | 前端已调用，只补后端 |
| DD5 | 前端WebSocket使用useWebSocket hook封装 | 集中管理连接状态 |

## 文件变更清单

| 文件 | 动作 | 说明 |
|------|------|------|
| `backend/app/model/lora_adapter.py` | modify | 新增DoraAdapter类(magnitude+归一化) |
| `backend/app/services/train/train_service.py` | modify | 新增validate_galore_config函数 |
| `backend/app/services/rag/retriever.py` | modify | 新增HippoRetriever类(PPR) |
| `backend/app/api/v1/chat.py` | modify | 新增GET /sessions端点 |
| `backend/app/services/chat/chat_service.py` | modify | 新增list_sessions方法 |
| `backend/app/schemas/chat.py` | modify | 新增ConversationItem/ListResponse schema |

## 关键接口

### DoraAdapter
- `__init__(linear, rank=8, alpha=None, freeze_base=True)`: magnitude初始化为base_weight逐行范数
- `_effective_weight()`: 归一化LoRA方向 * magnitude
- `forward(x)`: 使用effective_weight计算
- `param_groups(base_lr)`: 返回lora_A/lora_B/magnitude三组

### validate_galore_config
- 输入: config dict → 输出: dict(use_galore/rank/update_proj_gap/scale)
- rank<1时raise ValueError

### HippoRetriever
- `retrieve(query, kb_id, top_k, metadata_filter)`: PPR重排序返回结果

### GET /api/v1/chat/sessions
- Auth: JWT | Query: page, page_size, status | Response: conversations + pagination
