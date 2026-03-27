# 研究发现 - Tri-Transformer 前端

## 背景
全新项目，无既有代码库。需从零搭建 React + TypeScript + Ant Design Pro 前端。

## 关键发现

### 发现 1: 项目为全新空项目
- **来源**: ls /Users/hyx/codespace/Tri-Transformer
- **内容**: 仅有 docs/ 目录，无 package.json、无 src/
- **影响**: 需从 Vite scaffold 开始，完整搭建项目

### 发现 2: PRD 明确指定技术栈
- **来源**: PRD 4.2 技术栈选型
- **内容**: React + Ant Design Pro 前端；FastAPI 后端；Milvus/Chroma 向量数据库
- **影响**: 前端框架已确定，无需评估替代方案

### 发现 3: 前端主要功能模块（来自 PRD 3.1.4）
- **内容**:
  1. 对话界面 - 单/多轮对话、知识来源追溯、历史保存/导出、文档上传提问
  2. 结果后处理展示 - 事实校验、格式标准化
  3. 可视化管理面板 - 知识库管理、训练监控、性能指标可视化
- **影响**: 三大页面/模块，可独立并行开发

### 发现 4: 性能目标
- **来源**: PRD 3.2.1
- **内容**: 检索 < 500ms，生成 < 2s，10+ 并发用户
- **影响**: 前端需做 loading 状态管理，避免 UI 阻塞

### 发现 5: 后端 API 未实现
- **来源**: 项目为全新空项目
- **内容**: 无后端代码，API 需自行约定
- **影响**: 需用 MSW（Mock Service Worker）模拟所有 API，前后端并行开发

### 发现 6: 部署要求 Docker Compose
- **来源**: PRD 3.2.5 + 路线图第一阶段
- **影响**: 前端需提供 Dockerfile + nginx 配置，集成到 docker-compose.yml

## 技术笔记

### API 约定设计
```
Base URL: http://localhost:8000/api/v1

认证:
  POST /auth/login    { username, password } → { token, user }
  POST /auth/register { username, password, email } → { user }
  POST /auth/logout

对话:
  GET  /conversations              → [{ id, title, createdAt }]
  POST /conversations              { title? } → { id }
  GET  /conversations/:id/messages → [{ role, content, sources, createdAt }]
  POST /conversations/:id/messages { content } → { message, sources }

文档:
  GET    /documents          → [{ id, name, type, status, createdAt }]
  POST   /documents/upload   FormData(file) → { id, status }
  DELETE /documents/:id
  POST   /documents/search   { query, topK? } → [{ doc, chunk, score }]

训练:
  GET /training/status  → { phase, progress, eta }
  GET /metrics          → { retrievalAccuracy, bleuScore, hallucinationRate, history[] }
```

### 图表库选择
- Recharts：React 原生，API 简洁，bundle 小
- @ant-design/charts：AntD 生态，但 bundle 大
- **选择 Recharts**：轻量，与 AntD 不冲突

## 参考资料
- PRD: docs/Tri-Transformer 可控对话与 RAG 知识库增强系统.md
- Ant Design Pro: https://pro.ant.design
- Zustand: https://zustand-demo.pmnd.rs
- MSW: https://mswjs.io
- Recharts: https://recharts.org
