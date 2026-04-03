# Tri-Transformer

AI 幻觉检测与实时通信系统 - 基于三分支 Transformer 架构（ITransformer / CTransformer / OTransformer）

[![CI/CD](https://github.com/your-org/tri-transformer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/tri-transformer/actions/workflows/ci-cd.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/react-18+-blue.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📋 目录

- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [开发指南](#-开发指南)
- [项目结构](#-项目结构)
- [模型训练与评估](#-模型训练与评估)
- [API 文档](#-api-文档)
- [测试](#-测试)
- [部署](#-部署)
- [贡献](#-贡献)

## ✨ 功能特性

- 🤖 **AI 幻觉检测**: 基于 PyTorch I/C/O 三分支 Transformer 模型，高精度检测 AI 生成内容中的幻觉
- 📚 **RAG 知识库问答**: 支持文档上传、向量化检索、混合检索 + BM25 重排序
- 💬 **实时通信**: WebSocket/WebRTC 实时音视频通信，支持文本/音频/视频三种对话模式
- 📊 **数据可视化**: React + Recharts 实时展示对话指标、训练状态等数据
- 🔐 **用户认证**: JWT 认证系统，支持用户注册、登录
- 🎯 **模型训练**: 支持 LoRA/QLoRA 微调，自定义损失函数（幻觉检测 / RAG 对齐 / 控制对齐）
- 🔬 **模型评估**: 完整评估管道，含 evaluate.py / evaluation.py / quick_start.py
- 🖥️ **推理 CLI**: 命令行推理工具 inference_cli.py，支持批量推理与流式输出
- 🔄 **流式响应**: 支持流式输出，实时显示生成内容
- 🚀 **CI/CD**: GitHub Actions 自动化构建、测试与部署

## 🛠️ 技术栈

### 前端
- **框架**: React 18 + TypeScript + Vite
- **UI 组件**: Ant Design 5
- **状态管理**: Zustand
- **图表**: Recharts
- **HTTP 客户端**: Axios
- **测试**: Vitest + Testing Library

### 后端
- **框架**: FastAPI (Python 3.10+)
- **数据库**: SQLAlchemy (Async) + SQLite/PostgreSQL
- **认证**: JWT (python-jose)
- **ML 框架**: PyTorch
- **RAG**: ChromaDB + Sentence Transformers + BM25
- **测试**: pytest + pytest-cov

### 基础设施
- **容器化**: Docker + Docker Compose
- **向量数据库**: Milvus / ChromaDB
- **CI/CD**: GitHub Actions

## 🚀 快速开始

### 前置要求

- Python 3.10+
- Node.js 20+
- pnpm (推荐) 或 npm
- Docker (可选)

### 1. 克隆项目

```bash
git clone https://github.com/your-org/tri-transformer.git
cd tri-transformer
```

### 2. 安装依赖

**Windows (PowerShell):**
```powershell
.\scripts\dev.ps1 install
```

**Linux/macOS:**
```bash
chmod +x scripts/dev.sh
./scripts/dev.sh install
```

### 3. 配置环境变量

**后端:**
```bash
cd backend
cp .env.example .env
# 编辑 .env 文件，配置 SECRET_KEY 等
```

**前端:**
```bash
cd frontend
cp .env.example .env
# 编辑 .env 文件 (通常使用默认值即可)
```

### 4. 启动开发服务器

**方式一：使用自动化脚本 (推荐)**

Windows:
```powershell
.\scripts\dev.ps1 dev
```

Linux/macOS:
```bash
./scripts/dev.sh dev
```

**方式二：手动启动**

后端:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

前端:
```bash
cd frontend
pnpm dev
```

### 5. 访问应用

- **前端**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 6. Docker 部署 (可选)

```bash
docker-compose up -d
```

访问:
- 前端：http://localhost:3002
- 后端：http://localhost:8002
- Milvus: http://localhost:19532

## 📖 开发指南

### 可用命令

**开发:**
```bash
# 安装所有依赖
./scripts/dev.sh install  # 或 .\scripts\dev.ps1 install

# 启动开发服务器
./scripts/dev.sh dev

# 运行测试
./scripts/dev.sh test

# 代码检查
./scripts/dev.sh lint

# 构建项目
./scripts/dev.sh build

# 清理构建文件
./scripts/dev.sh clean
```

**前端单独命令:**
```bash
cd frontend
pnpm dev          # 开发服务器
pnpm test         # 运行测试
pnpm lint         # ESLint 检查
pnpm typecheck    # TypeScript 类型检查
pnpm build        # 生产构建
```

**后端单独命令:**
```bash
cd backend
uvicorn app.main:app --reload  # 开发服务器
pytest                         # 运行测试
black app/ tests/              # 代码格式化
flake8 app/ tests/             # Lint 检查
```

### 项目结构

```
tri-transformer/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── api/            # API 路由
│   │   ├── core/           # 核心配置
│   │   ├── models/         # 数据模型
│   │   ├── schemas/        # Pydantic schemas
│   │   └── services/       # 业务逻辑
│   ├── tests/              # 测试文件
│   └── requirements.txt    # Python 依赖
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── api/           # API 客户端
│   │   ├── components/    # React 组件
│   │   ├── hooks/         # Custom hooks
│   │   ├── layouts/       # 布局组件
│   │   ├── pages/         # 页面组件
│   │   ├── store/         # Zustand stores
│   │   └── utils/         # 工具函数
│   ├── tests/             # 测试文件
│   └── package.json       # Node 依赖
├── eval/                   # 评估管道
│   ├── loss/              # 损失函数
│   ├── pipeline/          # 评估流程
│   └── tests/             # 测试文件
├── docs/                   # 文档
│   ├── agent/             # 开发指南（架构/命令/测试/约定）
│   ├── installation/      # 安装指南
│   ├── API_REFERENCE.md   # API 参考手册
│   ├── FAQ.md             # 常见问题
│   └── QUICKSTART.md      # 快速入门
├── scripts/                # 自动化脚本
│   ├── dev.ps1           # Windows 脚本
│   ├── dev.sh            # Linux/macOS 脚本
│   ├── check-status.ps1  # Windows 状态检查
│   └── check-status.sh   # Linux/macOS 状态检查
└── docker-compose.yml     # Docker 配置
```

## 🔬 模型训练与评估

### 快速验证模型

```bash
cd backend
python verify_model.py
```

### 训练模型

```bash
cd backend
python app/services/model/train.py --config configs/train_config.yaml
```

### 评估模型

```bash
cd backend
python app/services/model/evaluate.py --checkpoint path/to/checkpoint
```

### 命令行推理

```bash
cd backend
python app/services/model/inference_cli.py --input "你的输入文本"
```

### 快速体验 Demo

```bash
python demo.py
```

详见 [训练指南](docs/agent/training_guide.md) 和 [模型实现总结](MODEL_IMPLEMENTATION_SUMMARY.md)。

## 📡 API 文档

启动后端后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 完整 API 参考: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

### 主要 API 端点

**认证:**
- `POST /api/v1/auth/register` - 用户注册
- `POST /api/v1/auth/login` - 用户登录

**对话:**
- `GET /api/v1/chat/sessions` - 获取会话列表
- `POST /api/v1/chat/sessions` - 创建会话
- `POST /api/v1/chat/sessions/{id}/messages` - 发送消息

**知识库:**
- `POST /api/v1/knowledge/documents` - 上传文档
- `GET /api/v1/knowledge/documents` - 获取文档列表
- `DELETE /api/v1/knowledge/documents/{id}` - 删除文档

**模型:**
- `POST /api/v1/model/infer` - 推理请求
- `GET /api/v1/model/status` - 模型状态

**训练:**
- `POST /api/v1/train/jobs` - 创建训练任务
- `GET /api/v1/train/jobs` - 获取训练任务列表
- `GET /api/v1/train/jobs/{id}` - 获取训练状态

## 🧪 测试

### 运行所有测试

```bash
./scripts/dev.sh test
```

### 单独运行

**后端测试:**
```bash
cd backend
pytest                    # 运行所有测试
pytest -v                 # 详细输出
pytest tests/test_chat.py # 指定测试文件
pytest --cov=app          # 覆盖率报告
```

**前端测试:**
```bash
cd frontend
pnpm test                 # 运行所有测试
pnpm test:watch          # 监听模式
```

**Eval 测试:**
```bash
cd eval
python -m pytest tests/ -v
```

## 🚢 部署

### 生产环境配置

1. 修改 `.env` 文件中的配置:
   - `SECRET_KEY` - 使用强随机密钥
   - `DATABASE_URL` - 使用 PostgreSQL
   - `DEBUG=False`
   - 配置正确的 CORS_ORIGINS

2. 构建 Docker 镜像:
```bash
docker-compose build
```

3. 启动服务:
```bash
docker-compose up -d
```

### 环境变量

详见 [backend/.env.example](backend/.env.example) 和 [frontend/.env.example](frontend/.env.example)

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md) 了解详细信息。

### 开发流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 后端：遵循 PEP 8，使用 black 格式化
- 前端：遵循 ESLint 规则，使用 TypeScript 严格模式

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- Transformer 架构的提出者
- FastAPI、React、PyTorch 等开源框架的贡献者
- 所有参与本项目的开发者

## 📚 相关文档

- [快速入门](docs/QUICKSTART.md)
- [安装指南](docs/installation/INSTALLATION.md)
- [API 参考](docs/API_REFERENCE.md)
- [常见问题](docs/FAQ.md)
- [更新日志](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md)
- [模型实现总结](MODEL_IMPLEMENTATION_SUMMARY.md)

## 📞 联系方式

如有问题或建议，请通过以下方式联系:
- GitHub Issues: [提交 issue](https://github.com/your-org/tri-transformer/issues)
- Email: your-email@example.com
