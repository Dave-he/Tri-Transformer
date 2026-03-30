# 开发命令参考

## 前端（React + Vite）

| 命令 | 说明 |
|------|------|
| `cd frontend && pnpm dev` | 启动 Vite 开发服务器（端口 3000） |
| `cd frontend && pnpm build` | TypeScript 编译 + Vite 生产构建 |
| `cd frontend && pnpm preview` | 预览生产构建 |
| `cd frontend && pnpm test` | 运行 Vitest 测试（单次） |
| `cd frontend && pnpm test:watch` | Vitest 监听模式 |
| `cd frontend && pnpm lint` | ESLint 检查（src 目录，0 warnings） |
| `cd frontend && pnpm typecheck` | TypeScript 类型检查（不生成文件） |

## 后端（FastAPI + Python）

| 命令 | 说明 |
|------|------|
| `cd backend && uvicorn app.main:app --reload` | 启动开发服务器（热重载，端口 8000） |
| `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000` | 生产启动 |
| `cd backend && pytest` | 运行所有测试 |
| `cd backend && pytest -v` | 详细测试输出 |
| `cd backend && pytest tests/test_chat.py` | 指定测试文件 |
| `cd backend && pytest --cov=app` | 测试覆盖率报告 |
| `cd backend && black app/ tests/` | 代码格式化（行宽 100） |
| `cd backend && flake8 app/ tests/` | Lint 检查 |

## Eval Pipeline

| 命令 | 说明 |
|------|------|
| `cd eval && python -m pytest` | 运行 eval 测试 |
| `cd eval && python scripts/run_eval.py` | 执行评估 |
| `cd eval && python scripts/ci_check.py` | CI 门禁检查 |

## Docker

| 命令 | 说明 |
|------|------|
| `docker-compose up -d` | 后台启动所有服务 |
| `docker-compose up --build` | 重新构建并启动 |
| `docker-compose down` | 停止并移除容器 |
| `docker-compose logs -f` | 实时查看日志 |
| `docker-compose ps` | 查看容器状态 |

## 依赖管理

```bash
# 前端
cd frontend && pnpm install           # 安装依赖
cd frontend && pnpm add <package>     # 添加依赖
cd frontend && pnpm add -D <package>  # 添加开发依赖

# 后端
pip install -r backend/requirements.txt
pip install -r eval/requirements.txt
```
