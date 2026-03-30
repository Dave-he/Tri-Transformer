# 开发规范

## 代码风格

### 前端（TypeScript/React）
- 组件使用函数式组件 + Hooks
- Props 类型使用 interface 定义
- 文件命名：组件 PascalCase，工具函数 camelCase
- 导入顺序：React → 第三方 → 内部模块 → 样式
- 禁止 `any` 类型，使用 `unknown` + 类型收窄
- 测试框架：Vitest + @testing-library/react

### 后端（Python/FastAPI）
- 遵循 PEP8，行宽 100
- 类型注解全覆盖（Python 3.10+）
- Pydantic v2 schemas 用于请求/响应验证
- 依赖注入通过 FastAPI Depends
- 异步优先（async/await）
- 测试框架：pytest + pytest-asyncio

## 目录约定

### 前端组件
- 通用组件：`frontend/src/components/`
- 页面组件：`frontend/src/pages/`
- Hooks：`frontend/src/hooks/`
- API 封装：`frontend/src/api/`
- 类型定义：`frontend/src/types/`

### 后端模块
- API 路由：`backend/app/api/`
- 业务服务：`backend/app/services/`
- 数据模型（DB）：`backend/app/models/`
- 数据模式（Pydantic）：`backend/app/schemas/`
- 核心配置：`backend/app/core/`

## Git 提交规范

```
feat: 新功能
fix: 问题修复
refactor: 代码重构
test: 测试相关
docs: 文档更新
chore: 构建/工具配置
```

## 测试规范

### 前端测试
```bash
cd frontend && pnpm test          # 单次运行
cd frontend && pnpm test:watch    # 监听模式
```
- 测试文件放在 `frontend/src/test/` 或同目录 `*.test.tsx`
- 覆盖率目标：语句 80%+

### 后端测试
```bash
cd backend && pytest              # 运行所有测试
cd backend && pytest -v           # 详细输出
cd backend && pytest tests/path   # 指定路径
```
- 测试文件放在 `backend/tests/`
- 使用 pytest fixtures 管理测试数据
