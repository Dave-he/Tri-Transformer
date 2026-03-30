# 研发资产报告

> 生成时间：2026-03-30
> 项目：Tri-Transformer（全栈 — React 前端 + FastAPI 后端）

## 项目概览

| 属性 | 值 |
|------|-----|
| 项目名称 | Tri-Transformer |
| 类型 | 全栈（React + FastAPI） |
| 前端框架 | React 18 + TypeScript + Vite |
| 后端框架 | FastAPI (Python 3.10) + SQLAlchemy |
| 核心功能 | Tri-Transformer 幻觉检测 + RAG 知识库 + WebRTC 实时通信 |
| 测试框架 | 前端: Vitest / 后端: pytest |

## 前端代码资产

### 组件（15个）
- `frontend/src/components/chat/` — 聊天相关组件（ChatInput, MessageList, SourcePanel, ChatModeTabs, AudioVisualizer, WebRTCControls）
- `frontend/src/components/training/` — 训练配置（TrainingConfigForm, ModelPluginSelector）
- `frontend/src/components/documents/` — 文档管理（DocumentList, UploadPanel, SearchTestPanel）
- `frontend/src/components/metrics/` — 指标展示（MetricsChart, TrainingStatusCard）

### 页面（6个）
- `frontend/src/pages/ChatPage.tsx` — 主聊天页面（RAG + WebRTC 模式）
- `frontend/src/pages/TrainingPage.tsx` — 模型训练页面
- `frontend/src/pages/DocumentsPage.tsx` — 知识库文档管理页
- `frontend/src/pages/MetricsPage.tsx` — 训练指标监控页
- `frontend/src/pages/LoginPage.tsx` — 登录页
- `frontend/src/pages/RegisterPage.tsx` — 注册页

### Hooks（3个）
- `frontend/src/hooks/useConversation.ts` — 对话管理
- `frontend/src/hooks/useDocuments.ts` — 文档操作
- `frontend/src/hooks/useAuth.ts` — 认证状态

### 状态管理（6个 Zustand stores）
- `frontend/src/store/conversationStore.ts` — 对话状态
- `frontend/src/store/documentStore.ts` — 文档状态
- `frontend/src/store/authStore.ts` — 用户认证
- `frontend/src/store/metricsStore.ts` — 训练指标
- `frontend/src/store/trainingConfigStore.ts` — 训练配置
- `frontend/src/store/webrtcStore.ts` — WebRTC 连接状态

### API 封装（6个）
- `frontend/src/api/conversations.ts` — 对话 API
- `frontend/src/api/documents.ts` — 知识库文档 API
- `frontend/src/api/auth.ts` — 认证 API
- `frontend/src/api/training.ts` — 训练任务 API
- `frontend/src/api/trainingConfig.ts` — 训练配置 API
- `frontend/src/api/webrtc.ts` — WebRTC 信令 API
- `frontend/src/api/client.ts` — Axios 基础客户端

### Mock 数据（7个 handlers）
- `frontend/src/mocks/handlers/` — MSW handlers（auth, conversations, documents, training, trainingConfig, webrtc）

## 后端代码资产

### API 路由（6个）
- `backend/app/api/v1/auth.py` — 注册/登录/JWT
- `backend/app/api/v1/chat.py` — RAG 对话
- `backend/app/api/v1/knowledge.py` — 知识库文档管理
- `backend/app/api/v1/model.py` — 模型信息/推理
- `backend/app/api/v1/train.py` — 训练任务管理
- `backend/app/api/v1/stream.py` — 流式输出（SSE/WebSocket）

### 服务层
- `backend/app/services/model/inference_service.py` — Tri-Transformer 推理服务
- `backend/app/services/model/fact_checker.py` — 事实核查服务
- `backend/app/services/model/stream_engine.py` — 流式生成引擎
- `backend/app/services/rag/document_processor.py` — 文档处理器
- `backend/app/services/rag/vector_store.py` — 向量存储
- `backend/app/services/rag/embedder.py` — 文本嵌入
- `backend/app/services/rag/retriever.py` — 检索器
- `backend/app/services/rag/reranker.py` — 重排序器
- `backend/app/services/chat/chat_service.py` — 对话服务
- `backend/app/services/train/train_service.py` — 训练服务

### 模型层（Tri-Transformer）
- `backend/app/model/tri_transformer.py` — 主模型架构
- `backend/app/model/branches.py` — 三分支（Branch1/2/3）
- `backend/app/model/lora_adapter.py` — LoRA 适配器
- `backend/app/model/pluggable_llm.py` — 可插拔 LLM
- `backend/app/model/trainer.py` — 模型训练器
- `backend/app/model/tokenizer/` — 统一分词器

### 数据模型（SQLAlchemy）
- `backend/app/models/user.py` — User（id, username, email, kb_id）
- `backend/app/models/document.py` — Document（知识库文档）
- `backend/app/models/chat_session.py` — ChatSession
- `backend/app/models/train_job.py` — TrainJob

### Pydantic Schemas
- `backend/app/schemas/auth.py` — 认证请求/响应
- `backend/app/schemas/chat.py` — 聊天请求/响应
- `backend/app/schemas/knowledge.py` — 知识库文档
- `backend/app/schemas/model.py` — 模型信息
- `backend/app/schemas/train.py` — 训练任务

## API 契约摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/auth/register | 用户注册 |
| POST | /api/v1/auth/login | 用户登录（JWT） |
| GET/POST | /api/v1/chat | RAG 对话 |
| GET/POST | /api/v1/knowledge | 知识库文档管理 |
| GET | /api/v1/model | 模型信息 |
| POST/GET | /api/v1/train | 训练任务管理 |
| WS/SSE | /api/v1/model/stream | 流式推理输出 |
| GET | /health | 健康检查 |

## 代码片段（.codeflicker/snippets/）

提取了 **22 个**代码片段：
- components: 15个
- stores: 2个
- views: 2个
- utils: 1个
- hooks: 1个
- other: 1个

## 缺口清单

### P1 建议
- 建议为后端 FastAPI 路由补充 OpenAPI 文档注释
- 建议配置 ESLint 配置文件（`frontend/eslint.config.ts`）

### P2 可选
- 考虑添加 E2E 测试（Playwright）
- 考虑 API 接口自动生成（openapi-typescript）
