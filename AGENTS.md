# Tri-Transformer

## WHY（项目目标）

Tri-Transformer 是一个基于三阶段 Transformer 架构的 AI 幻觉检测与实时通信系统，提供：
- 高精度幻觉检测（基于 PyTorch Tri-Transformer 模型）
- 实时 WebSocket 通信
- 直观的 React 数据可视化前端

## WHAT（技术栈）

| 层 | 技术 | 说明 |
|----|------|------|
| 前端 | React 18 + TypeScript + Vite | SPA，Ant Design UI |
| 状态管理 | Zustand | 轻量全局状态 |
| 图表 | Recharts | 数据可视化 |
| 后端 | FastAPI (Python 3.10) | REST API + WebSocket |
| 模型 | PyTorch + Tri-Transformer | 幻觉检测推理 |
| 容器 | Docker + docker-compose | 一键部署 |

## HOW（开发命令）

### 前端
```bash
cd frontend && pnpm dev          # 启动开发服务器
cd frontend && pnpm build        # 生产构建
cd frontend && pnpm test         # 运行测试 (Vitest)
cd frontend && pnpm lint         # ESLint 检查
cd frontend && pnpm typecheck    # TypeScript 类型检查
```

### 后端
```bash
cd backend && uvicorn app.main:app --reload   # 启动开发服务器
cd backend && pytest                          # 运行测试
cd backend && black app/ tests/              # 代码格式化
cd backend && flake8 app/ tests/             # Lint 检查
```

### Docker
```bash
docker-compose up -d     # 启动所有服务
docker-compose down      # 停止所有服务
docker-compose logs -f   # 查看日志
```

## 项目结构

```
Tri-Transformer/
├── frontend/             # React + Vite 前端
│   ├── src/
│   │   ├── api/          # API 请求封装
│   │   ├── components/   # 通用组件
│   │   ├── hooks/        # 自定义 Hooks
│   │   ├── layouts/      # 布局组件
│   │   ├── pages/        # 页面组件
│   │   ├── store/        # Zustand 状态管理
│   │   ├── types/        # TypeScript 类型定义
│   │   └── utils/        # 工具函数
│   └── package.json
├── backend/              # FastAPI Python 后端
│   ├── app/
│   │   ├── api/          # API 路由
│   │   ├── core/         # 核心配置
│   │   ├── model/        # Tri-Transformer 模型
│   │   ├── models/       # 数据库模型
│   │   ├── schemas/      # Pydantic 数据模式
│   │   └── services/     # 业务逻辑服务
│   ├── tests/            # 测试
│   └── requirements.txt
├── docs/                 # 文档
│   ├── agent/            # AI Flow 开发约定
│   ├── tasks/            # 任务文档
│   └── research/         # 研发资产
├── docker-compose.yml
└── .ai-flow.config.js
```

## AI Flow 规范

- 所有需求以 Task ID 形式提交
- 技术方案存放在 `docs/tasks/`
- 研发资产索引在 `docs/research/`
- 代码片段在 `.codeflicker/snippets/`
