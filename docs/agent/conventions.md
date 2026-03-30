# 开发规范

## 代码风格

### 前端（TypeScript/React）
- 函数式组件 + Hooks，禁止 class 组件
- Props 类型使用 `interface`，不用 `type`
- 文件命名：组件 PascalCase，工具函数 camelCase
- 导入顺序：React → 第三方库 → `@/` 内部模块 → 样式
- 禁止 `any`，使用 `unknown` + 类型收窄
- `@/` path alias 指向 `frontend/src/`

### 后端（Python/FastAPI）
- PEP8，行宽 100（black 格式化）
- 类型注解全覆盖（Python 3.10+ syntax）
- Pydantic v2 schemas 用于请求/响应验证
- 依赖注入通过 `FastAPI Depends()`，不直接实例化服务
- 全异步 (async/await)，禁止同步阻塞 I/O

### Eval 模块（Python）
- 自定义损失函数继承 `nn.Module`，实现 `forward()` 方法
- 评估器通过 `EvalPipeline` 组合，不直接调用单个评估器
- CI 门禁阈值配置在 `eval/pipeline/ci_gate.py`

## 目录约定

### 前端
| 目录 | 用途 |
|------|------|
| `frontend/src/api/` | axios 封装，每个功能域一个文件 |
| `frontend/src/components/` | UI 组件，按功能域子目录分组 |
| `frontend/src/pages/` | 页面组件，与路由一一对应 |
| `frontend/src/store/` | Zustand stores，每个功能域一个文件 |
| `frontend/src/hooks/` | 自定义 Hooks |
| `frontend/src/types/` | 共享 TypeScript 类型定义 |
| `frontend/src/mocks/handlers/` | MSW mock handlers，按功能域拆分 |

### 后端
| 目录 | 用途 |
|------|------|
| `backend/app/api/v1/` | FastAPI 路由（每个资源一个文件） |
| `backend/app/services/` | 业务逻辑（按功能域子目录） |
| `backend/app/models/` | SQLAlchemy ORM 模型 |
| `backend/app/schemas/` | Pydantic 请求/响应 schemas |
| `backend/app/model/` | PyTorch 模型代码 |
| `backend/app/core/` | 配置、DB 连接、安全工具 |

## Git 提交规范

```
feat: 新功能
fix: 问题修复
refactor: 代码重构
test: 测试相关
docs: 文档更新
chore: 构建/工具配置
```

## AI Flow 约定

- 需求以 Task ID 形式提交，技术方案存放在 `docs/tasks/<task-id>/`
- 研发资产索引在 `docs/research/rd-assets.md`
- 代码片段沉淀到 `.codeflicker/snippets/`
- 新增重要组件/服务后更新 `docs/research/rd-assets.md`
