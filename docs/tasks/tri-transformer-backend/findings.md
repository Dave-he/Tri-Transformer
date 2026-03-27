# 研究发现 - Tri-Transformer 后端

## 背景
基于 FastAPI/Python 开发 RAG 知识库增强系统后端，需要构建 RAG 流水线、对话管理、推理接口等模块。

## 关键发现

### 发现 1: PRD 技术栈明确指定
- **来源**: PRD 第 4.2 节
- **内容**: 后端 FastAPI/Python，RAG 引擎 LlamaIndex/LangChain，向量库 Milvus/Chroma，嵌入 BGE 系列，重排 BGE Reranker
- **影响**: 技术选型无争议，直接按 PRD 实现

### 发现 2: MVP 阶段使用 Chroma
- **来源**: PRD 第 5.1 节路线图
- **内容**: MVP 使用 Chroma（个人），企业版使用 Milvus
- **影响**: 本次开发以 Chroma 为主，代码需设计成可替换架构

### 发现 3: 推理接口需要 Mock 支持
- **来源**: 需求分析
- **内容**: Tri-Transformer 是研究性模型，开发阶段无法保证 GPU 可用
- **影响**: 推理模块必须支持 mock_mode=True，返回模拟响应

### 发现 4: 文档处理使用 Unstructured + PyMuPDF
- **来源**: PRD 第 4.2 节
- **内容**: 文档处理库选型 Unstructured/PyMuPDF/PaddleOCR
- **影响**: 需在 requirements.txt 中声明，PDF 解析用 PyMuPDF

### 发现 5: 推理流程完整链路
- **来源**: PRD 第 4.4 节
- **内容**: 用户输入 → RAG 检索 → 融合 → I/C/O 分支推理 → 后处理 → 返回
- **影响**: 对话 API 需串联完整链路，每步需有耗时监控

## 技术笔记

### 项目目录结构规划
```
backend/
├── app/
│   ├── main.py                # FastAPI 应用入口
│   ├── core/
│   │   ├── config.py          # 配置管理
│   │   ├── database.py        # DB 连接
│   │   └── security.py        # JWT 工具
│   ├── api/
│   │   └── v1/
│   │       ├── auth.py        # 认证路由
│   │       ├── chat.py        # 对话路由
│   │       ├── knowledge.py   # 知识库路由
│   │       ├── model.py       # 推理路由
│   │       └── train.py       # 训练调度路由
│   ├── models/                # SQLAlchemy ORM 模型
│   ├── schemas/               # Pydantic 模式
│   ├── services/
│   │   ├── rag/               # RAG 引擎服务
│   │   ├── chat/              # 对话管理服务
│   │   ├── model/             # 推理服务（含 mock）
│   │   └── train/             # 训练调度服务
│   └── dependencies.py        # FastAPI 依赖注入
├── tests/
│   ├── test_auth.py
│   ├── test_chat.py
│   ├── test_knowledge.py
│   ├── test_rag.py
│   └── test_model.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 核心 API 端点规划
- POST /api/v1/auth/register
- POST /api/v1/auth/login
- POST /api/v1/knowledge/documents
- GET  /api/v1/knowledge/documents
- DELETE /api/v1/knowledge/documents/{id}
- GET /api/v1/knowledge/search
- POST /api/v1/chat/sessions
- POST /api/v1/chat/sessions/{id}/messages
- GET  /api/v1/chat/sessions/{id}/history
- POST /api/v1/model/inference
- POST /api/v1/train/jobs
- GET  /api/v1/train/jobs/{id}
