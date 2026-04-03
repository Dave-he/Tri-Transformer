# 快速启动指南 - Tri-Transformer

## 🚀 5 分钟快速开始

### 步骤 1: 检查项目状态

**Windows:**
```powershell
.\scripts\check-status.ps1
```

**Linux/macOS:**
```bash
chmod +x scripts/check-status.sh
./scripts/check-status.sh
```

### 步骤 2: 配置环境变量

**后端:**
```bash
cd backend
cp .env.example .env
```

编辑 `.env` 文件，至少修改:
```env
SECRET_KEY=your-random-secret-key-here
JWT_SECRET_KEY=your-random-jwt-secret-here
```

**前端:**
```bash
cd frontend
cp .env.example .env
```

通常使用默认值即可。

### 步骤 3: 安装依赖

**Windows (PowerShell):**
```powershell
.\scripts\dev.ps1 install
```

**Linux/macOS:**
```bash
chmod +x scripts/dev.sh
./scripts/dev.sh install
```

### 步骤 4: 启动开发服务器

**方式一：自动化脚本 (推荐)**

**Windows:**
```powershell
.\scripts\dev.ps1 dev
```

**Linux/macOS:**
```bash
./scripts/dev.sh dev
```

**方式二：手动启动**

后端:
```bash
cd backend
uvicorn app.main:app --reload
```

前端:
```bash
cd frontend
pnpm dev
```

### 步骤 5: 访问应用

- **前端**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 📝 常用命令

### 开发
```bash
# 安装依赖
./scripts/dev.sh install

# 启动开发服务器
./scripts/dev.sh dev

# 检查项目状态
./scripts/check-status.sh
```

### 测试
```bash
# 运行所有测试
./scripts/dev.sh test

# 仅后端测试
cd backend && pytest

# 仅前端测试
cd frontend && pnpm test
```

### 代码质量
```bash
# 后端格式化
cd backend && black app/ tests/

# 后端 lint
cd backend && flake8 app/ tests/

# 前端 lint
cd frontend && pnpm lint

# 前端类型检查
cd frontend && pnpm typecheck
```

### 构建和部署
```bash
# 构建前端
cd frontend && pnpm build

# Docker 部署
docker-compose up -d
```

## 🔧 常见问题

### Q1: npm install 失败

**解决方案:**
```bash
npm cache clean --force
npm install
```

或使用 pnpm:
```bash
npm install -g pnpm
pnpm install
```

### Q2: 后端依赖安装失败 (numpy 编译错误)

**原因**: Windows 缺少 C 编译器

**解决方案**:
1. 使用预编译的 wheel 包 (已配置)
2. 或安装 Visual Studio Build Tools

### Q3: 端口被占用

**解决方案**: 修改启动命令的端口
```bash
# 后端
uvicorn app.main:app --reload --port 8001

# 前端 (修改 vite.config.ts)
```

### Q4: 数据库文件不存在

**解决方案**: 首次启动时会自动创建
```bash
cd backend
# 确保 tritransformer.db 目录存在
```

## 📚 下一步

1. **查看 API 文档**: http://localhost:8000/docs
2. **阅读完整文档**: [README.md](README.md)
3. **了解贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. **查看项目架构**: [docs/agent/architecture.md](docs/agent/architecture.md)

## 🆘 获取帮助

- 查看 [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md) 了解项目状态
- 提交 Issue: https://github.com/your-org/tri-transformer/issues
- 查看文档: [docs/](docs/)

---

**提示**: 运行 `./scripts/check-status.sh` 快速诊断项目状态!
