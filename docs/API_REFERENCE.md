# API 参考文档

> Tri-Transformer 系统完整 API 接口说明

本文档提供 Tri-Transformer 后端 API 的完整参考，包括认证、对话、知识库、模型推理等所有接口。

---

## 📑 目录

- [API 概览](#api-概览)
- [认证接口](#认证接口)
- [对话接口](#对话接口)
- [知识库接口](#知识库接口)
- [模型接口](#模型接口)
- [训练接口](#训练接口)
- [流式接口](#流式接口)
- [错误码说明](#错误码说明)

---

## API 概览

### 基础信息

- **基础 URL**: `http://localhost:8000/api/v1`
- **认证方式**: JWT Bearer Token
- **数据格式**: JSON
- **API 文档**: 访问 http://localhost:8000/docs 查看交互式 Swagger 文档

### 认证流程

```bash
# 1. 登录获取 Token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# 2. 在请求中使用 Token
curl -X GET "http://localhost:8000/api/v1/chat/sessions" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## 认证接口

### 1. 用户注册

**端点**: `POST /api/v1/auth/register`

**请求**:

```json
{
  "username": "string (required, 3-50 chars)",
  "email": "string (required, email format)",
  "password": "string (required, min 8 chars)",
  "full_name": "string (optional)"
}
```

**响应**:

```json
{
  "id": "uuid",
  "username": "string",
  "email": "string",
  "created_at": "datetime",
  "is_active": true
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

---

### 2. 用户登录

**端点**: `POST /api/v1/auth/login`

**请求**:

```json
{
  "username": "string",
  "password": "string"
}
```

**响应**:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string"
  }
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Admin123!@#"}'
```

---

### 3. 获取当前用户信息

**端点**: `GET /api/v1/auth/me`

**认证**: Required

**响应**:

```json
{
  "id": "uuid",
  "username": "string",
  "email": "string",
  "full_name": "string",
  "created_at": "datetime",
  "is_active": true,
  "kb_id": "uuid"
}
```

**示例**:

```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

### 4. 刷新 Token

**端点**: `POST /api/v1/auth/refresh`

**请求**:

```json
{
  "refresh_token": "string"
}
```

**响应**:

```json
{
  "access_token": "new_access_token",
  "expires_in": 1800
}
```

---

## 对话接口

### 1. 创建对话会话

**端点**: `POST /api/v1/chat/sessions`

**认证**: Required

**请求**:

```json
{
  "title": "string (optional)",
  "mode": "rag|chat|hallucination_detection",
  "model_config": {
    "enable_thinking": false,
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

**响应**:

```json
{
  "id": "uuid",
  "title": "string",
  "mode": "rag",
  "created_at": "datetime",
  "updated_at": "datetime",
  "message_count": 0
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "技术文档问答",
    "mode": "rag",
    "model_config": {
      "enable_thinking": true,
      "temperature": 0.5
    }
  }'
```

---

### 2. 发送消息

**端点**: `POST /api/v1/chat/sessions/{session_id}/messages`

**认证**: Required

**路径参数**:
- `session_id`: UUID

**请求**:

```json
{
  "content": "string (required)",
  "mode": "rag|chat|hallucination_detection",
  "stream": false,
  "rag_config": {
    "top_k": 5,
    "use_rerank": true,
    "include_sources": true
  },
  "hallucination_config": {
    "threshold": 0.3,
    "check_facts": true
  }
}
```

**响应**:

```json
{
  "id": "uuid",
  "session_id": "uuid",
  "role": "assistant",
  "content": "string",
  "sources": [
    {
      "document_id": "uuid",
      "content": "string",
      "score": 0.95
    }
  ],
  "hallucination_score": 0.12,
  "created_at": "datetime"
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/chat/sessions/SESSION_ID/messages" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "请总结文档的主要内容",
    "mode": "rag",
    "rag_config": {
      "top_k": 10,
      "use_rerank": true
    }
  }'
```

---

### 3. 获取对话历史

**端点**: `GET /api/v1/chat/sessions/{session_id}/messages`

**认证**: Required

**查询参数**:
- `limit`: 整数，默认 50
- `offset`: 整数，默认 0

**响应**:

```json
{
  "total": 100,
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "string",
      "created_at": "datetime"
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "string",
      "sources": [...],
      "created_at": "datetime"
    }
  ]
}
```

---

### 4. 删除对话会话

**端点**: `DELETE /api/v1/chat/sessions/{session_id}`

**认证**: Required

**响应**: `204 No Content`

---

### 5. 列出所有会话

**端点**: `GET /api/v1/chat/sessions`

**认证**: Required

**查询参数**:
- `limit`: 整数，默认 20
- `offset`: 整数，默认 0
- `mode`: 可选，过滤模式

**响应**:

```json
{
  "total": 50,
  "sessions": [
    {
      "id": "uuid",
      "title": "string",
      "mode": "rag",
      "message_count": 25,
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ]
}
```

---

## 知识库接口

### 1. 上传文档

**端点**: `POST /api/v1/knowledge/documents`

**认证**: Required

**Content-Type**: `multipart/form-data`

**表单字段**:
- `file`: 文件（required）
- `title`: 字符串（required）
- `description`: 字符串（optional）
- `tags`: 字符串数组（optional）

**响应**:

```json
{
  "id": "uuid",
  "title": "string",
  "filename": "string",
  "file_size": 1024000,
  "status": "processing|indexed|failed",
  "created_at": "datetime"
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/documents" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@./document.pdf" \
  -F "title=产品手册" \
  -F "description=2024 版产品技术文档"
```

---

### 2. 列出文档

**端点**: `GET /api/v1/knowledge/documents`

**认证**: Required

**查询参数**:
- `limit`: 整数，默认 20
- `offset`: 整数，默认 0
- `status`: processing|indexed|failed
- `search`: 搜索关键词

**响应**:

```json
{
  "total": 100,
  "documents": [
    {
      "id": "uuid",
      "title": "string",
      "filename": "string",
      "file_size": 1024000,
      "status": "indexed",
      "created_at": "datetime",
      "indexed_at": "datetime"
    }
  ]
}
```

---

### 3. 获取文档详情

**端点**: `GET /api/v1/knowledge/documents/{document_id}`

**认证**: Required

**响应**:

```json
{
  "id": "uuid",
  "title": "string",
  "description": "string",
  "filename": "string",
  "file_size": 1024000,
  "status": "indexed",
  "chunks_count": 150,
  "tags": ["技术文档", "产品"],
  "metadata": {
    "pages": 50,
    "language": "zh-CN"
  },
  "created_at": "datetime",
  "indexed_at": "datetime"
}
```

---

### 4. 删除文档

**端点**: `DELETE /api/v1/knowledge/documents/{document_id}`

**认证**: Required

**响应**: `204 No Content`

---

### 5. 重新处理文档

**端点**: `POST /api/v1/knowledge/documents/{document_id}/reprocess`

**认证**: Required

**响应**:

```json
{
  "status": "processing",
  "message": "文档已加入重新处理队列"
}
```

---

### 6. 检索文档内容

**端点**: `POST /api/v1/knowledge/retrieve`

**认证**: Required

**请求**:

```json
{
  "query": "string (required)",
  "top_k": 5,
  "use_rerank": true,
  "filter": {
    "document_ids": ["uuid1", "uuid2"],
    "tags": ["技术文档"]
  }
}
```

**响应**:

```json
{
  "query": "string",
  "results": [
    {
      "document_id": "uuid",
      "document_title": "string",
      "chunk_id": "uuid",
      "content": "string",
      "score": 0.95,
      "metadata": {
        "page": 10,
        "section": "3.2"
      }
    }
  ],
  "total_time_ms": 125
}
```

**示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/retrieve" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "产品的主要功能",
    "top_k": 10,
    "use_rerank": true
  }'
```

---

## 模型接口

### 1. 获取模型信息

**端点**: `GET /api/v1/model`

**认证**: Required

**响应**:

```json
{
  "model_name": "Tri-Transformer",
  "version": "1.0.0",
  "branches": {
    "i_transformer": {
      "model": "Qwen3-8B",
      "layers": 36,
      "hidden_size": 4096
    },
    "c_transformer": {
      "model": "DiT-Control",
      "layers": 8,
      "state_slots": 16
    },
    "o_transformer": {
      "model": "Qwen3-8B",
      "layers": 36,
      "hidden_size": 4096
    }
  },
  "capabilities": [
    "rag",
    "hallucination_detection",
    "streaming",
    "thinking_mode"
  ],
  "device": "cuda",
  "dtype": "bfloat16"
}
```

---

### 2. 获取模型性能指标

**端点**: `GET /api/v1/model/metrics`

**认证**: Required

**响应**:

```json
{
  "inference": {
    "avg_latency_ms": 150,
    "tokens_per_second": 85.5,
    "gpu_memory_used_gb": 28.5,
    "gpu_utilization": 0.75
  },
  "rag": {
    "avg_retrieval_time_ms": 45,
    "avg_rerank_time_ms": 30
  },
  "cache": {
    "hit_rate": 0.65,
    "size_mb": 512
  }
}
```

---

### 3. 清空模型缓存

**端点**: `POST /api/v1/model/clear-cache`

**认证**: Required

**响应**:

```json
{
  "status": "success",
  "cleared_memory_mb": 512
}
```

---

## 训练接口

### 1. 创建训练任务

**端点**: `POST /api/v1/train/jobs`

**认证**: Required

**请求**:

```json
{
  "job_type": "lora_finetune|full_finetune|alignment",
  "dataset": {
    "dataset_id": "uuid",
    "split": "train",
    "validation_split": 0.1
  },
  "config": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "use_gradient_checkpointing": true,
    "use_flash_attention": true
  },
  "branches": {
    "train_i_branch": true,
    "train_c_branch": true,
    "train_o_branch": false,
    "freeze_pluggable_llm": true
  }
}
```

**响应**:

```json
{
  "id": "uuid",
  "status": "pending|running|completed|failed",
  "progress": 0,
  "created_at": "datetime",
  "estimated_duration_minutes": 120
}
```

---

### 2. 获取训练任务列表

**端点**: `GET /api/v1/train/jobs`

**认证**: Required

**查询参数**:
- `limit`: 整数，默认 20
- `status`: pending|running|completed|failed

**响应**:

```json
{
  "total": 50,
  "jobs": [
    {
      "id": "uuid",
      "job_type": "lora_finetune",
      "status": "running",
      "progress": 0.45,
      "current_epoch": 2,
      "current_step": 450,
      "created_at": "datetime",
      "started_at": "datetime"
    }
  ]
}
```

---

### 3. 获取训练任务详情

**端点**: `GET /api/v1/train/jobs/{job_id}`

**认证**: Required

**响应**:

```json
{
  "id": "uuid",
  "status": "running",
  "progress": 0.45,
  "current_epoch": 2,
  "total_epochs": 3,
  "current_step": 450,
  "total_steps": 1000,
  "metrics": {
    "train_loss": 0.234,
    "eval_loss": 0.256,
    "learning_rate": 8.5e-5,
    "gpu_memory_used_gb": 30.5
  },
  "logs": [
    "Epoch 1/3 completed",
    "Evaluation: loss=0.256"
  ],
  "created_at": "datetime",
  "started_at": "datetime",
  "estimated_finish": "datetime"
}
```

---

### 4. 取消训练任务

**端点**: `POST /api/v1/train/jobs/{job_id}/cancel`

**认证**: Required

**响应**:

```json
{
  "status": "cancelled",
  "message": "训练任务已取消"
}
```

---

### 5. 上传训练数据集

**端点**: `POST /api/v1/train/datasets`

**认证**: Required

**Content-Type**: `multipart/form-data`

**表单字段**:
- `file`: JSONL 文件（required）
- `name`: 字符串（required）
- `description`: 字符串（optional）
- `dataset_type`: instruction|dialogue|alignment

**响应**:

```json
{
  "id": "uuid",
  "name": "string",
  "file_size": 10485760,
  "samples_count": 5000,
  "status": "processing|ready",
  "created_at": "datetime"
}
```

---

## 流式接口

### 1. WebSocket 流式推理

**端点**: `WS /api/v1/model/stream`

**认证**: Required (通过 Query Parameter 或 Header)

**连接**:

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/model/stream?token=${YOUR_TOKEN}`
);

ws.onopen = () => {
  ws.send(JSON.stringify({
    "message": "请总结文档内容",
    "session_id": "uuid",
    "stream": true,
    "enable_thinking": false
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Token:", data.token);
  console.log("Is done:", data.done);
};
```

**消息格式**:

```json
{
  "token": "string",
  "token_id": 1234,
  "logprob": -0.123,
  "done": false,
  "finish_reason": null
}
```

**完成消息**:

```json
{
  "token": "",
  "done": true,
  "finish_reason": "stop",
  "total_tokens": 256,
  "total_time_ms": 2340
}
```

---

### 2. SSE 流式输出

**端点**: `GET /api/v1/model/stream-sse`

**认证**: Required

**查询参数**:
- `message`: 字符串
- `session_id`: UUID
- `stream`: true

**响应** (Server-Sent Events):

```
data: {"token":"三","done":false}

data: {"token":"角","done":false}

data: {"token":"形","done":false}

data: {"token":"","done":true,"finish_reason":"stop"}
```

---

## 错误码说明

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 204 | 删除成功（无内容） |
| 400 | 请求参数错误 |
| 401 | 未认证或 Token 过期 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 409 | 资源冲突 |
| 422 | 数据验证错误 |
| 429 | 请求频率超限 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

### 业务错误码

| 错误码 | 说明 |
|--------|------|
| ERR_AUTH_001 | 用户名或密码错误 |
| ERR_AUTH_002 | Token 已过期 |
| ERR_AUTH_003 | Token 无效 |
| ERR_CHAT_001 | 会话不存在 |
| ERR_CHAT_002 | 消息内容为空 |
| ERR_RAG_001 | 文档处理失败 |
| ERR_RAG_002 | 检索失败 |
| ERR_MODEL_001 | 模型加载失败 |
| ERR_MODEL_002 | 推理失败 |
| ERR_TRAIN_001 | 训练任务创建失败 |
| ERR_TRAIN_002 | 数据集格式错误 |

### 错误响应格式

```json
{
  "error": {
    "code": "ERR_CHAT_001",
    "message": "会话不存在",
    "details": {
      "session_id": "invalid-uuid"
    }
  }
}
```

---

## 速率限制

| 端点 | 限制 |
|------|------|
| /api/v1/auth/login | 10 次/分钟 |
| /api/v1/chat/sessions/{id}/messages | 60 次/分钟 |
| /api/v1/knowledge/retrieve | 30 次/分钟 |
| /api/v1/model/stream | 10 次/分钟 |
| 其他端点 | 100 次/分钟 |

---

## 最佳实践

### 1. 错误处理

```javascript
async function sendMessage(message) {
  try {
    const response = await fetch('/api/v1/chat/sessions/SESSION_ID/messages', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ content: message })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error.message);
    }
    
    return await response.json();
  } catch (error) {
    console.error('发送消息失败:', error);
    // 实现重试逻辑或用户提示
  }
}
```

### 2. Token 刷新

```javascript
async function refreshToken() {
  const response = await fetch('/api/v1/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      refresh_token: localStorage.getItem('refresh_token')
    })
  });
  
  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
  return data.access_token;
}
```

### 3. 流式处理

```javascript
async function* streamResponse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        yield data;
      }
    }
  }
}

// 使用
const response = await fetch('/api/v1/model/stream-sse?message=hello&stream=true');
for await (const data of streamResponse(response)) {
  console.log('Token:', data.token);
}
```

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team
