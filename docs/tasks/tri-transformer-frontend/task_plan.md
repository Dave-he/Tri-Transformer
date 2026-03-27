# 任务计划: Tri-Transformer 前端开发

## 目标
为 Tri-Transformer 可控对话与 RAG 知识库增强系统构建完整 Web 前端，包含：对话界面、文档上传、RAG 知识库管理面板、训练状态监控、性能指标可视化仪表盘、用户认证。

## 当前阶段
Phase 2: 技术方案设计

## 阶段规划

### Phase 1: 需求分析（Stage 0-1）
- [x] 读取 PRD 文档
- [x] 执行需求质量门禁
- [x] 生成 requirement.yaml + requirement.md
- **Status:** complete

### Phase 2: 技术方案设计（Stage 3）
- [ ] 定义项目结构（Vite + React + TypeScript）
- [ ] 定义路由架构（React Router）
- [ ] 定义状态管理方案（Zustand）
- [ ] 定义 API 层设计（Axios + Mock）
- [ ] 定义组件架构
- [ ] 生成 tech-solution.yaml + tech-solution.md
- **Status:** in_progress

### Phase 3: 任务拆解（Stage 5）
- [ ] 拆分任务为 TDD 任务单元
- [ ] 生成 plan.yaml + plan.md（test → code 依赖）
- **Status:** pending

### Phase 4: 方案验证（Stage 6）
- [ ] 校验 plan.yaml 与 tech-solution.yaml 一致性
- [ ] 生成 verification-report.md
- **Status:** pending

### Phase 5: TDD 实现（Stage 7）
关键模块：
- [ ] 项目初始化（Vite + React + TS + AntD）
- [ ] 路由与布局搭建
- [ ] 用户认证模块（登录/注册页 + AuthStore + API）
- [ ] 对话界面（ChatPanel + MessageList + InputBox + SourcePanel）
- [ ] 文档上传组件（UploadPanel + 进度状态）
- [ ] 知识库管理页（DocList + 上传/删除 + 检索测试）
- [ ] 训练监控页（TrainingStatus + MetricsDashboard + 图表）
- [ ] 全局状态（ConversationStore + DocumentStore + MetricsStore）
- [ ] API 客户端 + Mock（MSW）
- **Status:** pending

### Phase 6: 文件变更审查（Stage 7.6）
- [ ] 增量 lint/typecheck 检查
- [ ] 验证变更符合预期范围
- **Status:** pending

## 关键决策记录

| 决策 | 理由 |
|------|------|
| Vite 代替 CRA | 更快的构建速度，更好的 TS 支持 |
| Zustand 代替 Redux | 轻量、无 boilerplate、适合中型项目 |
| Ant Design 5.x | PRD 明确指定 Ant Design Pro |
| Vitest 代替 Jest | 与 Vite 原生集成，无需额外配置 |
| MSW Mock | 后端 API 未实现，前端先按约定 Schema Mock |
| React Router v6 | 最新稳定版，hooks-first API |
| Axios + interceptors | 统一 token 注入、错误处理 |
| Recharts 图表库 | 轻量、React 原生、开箱即用 |

## 错误记录

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| （待记录） | - | - |

## 核心关注点
1. 全新项目搭建，需从 Vite 初始化开始
2. 后端 API 未实现，需用 MSW 做完整 Mock
3. 流式生成（SSE）需特殊处理，先做轮询版本
4. 图表组件（Recharts/AntD Charts）需验证可用性
5. 文件上传需处理大文件分片、进度显示
