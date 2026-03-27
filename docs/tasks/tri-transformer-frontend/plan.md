# 任务清单 - Tri-Transformer 前端

> 来源：docs/tasks/tri-transformer-frontend/tech-solution.yaml
> 测试命令：`cd frontend && npx vitest run`
> 构建命令：`cd frontend && npm run build`

## 任务统计

| 类型 | P0 | P1 | 合计 |
|------|----|----|------|
| test | 10 | 2  | 12   |
| code | 11 | 3  | 14   |
| 合计 | 21 | 5  | 26   |

## 执行顺序（TDD 原则：test → code）

```
T0-1 → P0-1（项目初始化）
T1-1 → P0-2（类型定义）
T2-1 → P0-3（API 客户端）
T3-1 → P0-4（MSW Mock）
T4-1 → P0-5（认证模块）
T5-1 → P0-6（主布局路由）
T6-1 → P0-7（对话 Store）
T7-1 + T7-2 → P0-8（对话界面）
T8-1 → P1-1（导出工具）
T9-1 → P0-9 → T9-2 → P0-10（文档管理）
T10-1 → P1-2（指标监控）
P0-6 → P0-11（通用组件）
P0-1 → P1-3（部署配置）
```

---

## P0 任务（阻塞性，必须完成）

### T0-1 验证项目初始化配置正确性
- **类型**：test | **依赖**：无
- **测试文件**：`frontend/src/test/setup.test.ts`
- **验收**：npm run build 成功，TypeScript 通过，Vitest 可运行

### P0-1 初始化 Vite + React + TypeScript 项目
- **类型**：code | **依赖**：T0-1
- **文件**：package.json, vite.config.ts, tsconfig.json, index.html, src/main.tsx, src/App.tsx

### T1-1 验证 API 类型定义完整性
- **类型**：test | **依赖**：P0-1
- **测试文件**：`frontend/src/types/__tests__/api.test.ts`

### P0-2 定义 TypeScript 类型
- **类型**：code | **依赖**：T1-1
- **文件**：src/types/api.ts, src/types/store.ts

### T2-1 为 Axios 客户端编写单元测试
- **类型**：test | **依赖**：P0-2
- **测试文件**：`frontend/src/api/__tests__/client.test.ts`
- **验收**：token 注入、401 重定向、错误处理

### P0-3 实现 API 客户端层
- **类型**：code | **依赖**：T2-1
- **文件**：src/api/client.ts, auth.ts, conversations.ts, documents.ts, training.ts

### T3-1 验证 MSW Mock handlers
- **类型**：test | **依赖**：P0-3
- **测试文件**：`frontend/src/mocks/__tests__/handlers.test.ts`

### P0-4 实现 MSW Mock handlers
- **类型**：code | **依赖**：T3-1
- **文件**：src/mocks/handlers/*.ts, src/mocks/browser.ts, src/mocks/server.ts

### T4-1 为 authStore 编写单元测试
- **类型**：test | **依赖**：P0-4
- **测试文件**：`frontend/src/store/__tests__/authStore.test.ts`
- **验收**：login/logout 状态变更、持久化

### P0-5 实现 authStore 和认证页面
- **类型**：code | **依赖**：T4-1
- **文件**：src/store/authStore.ts, hooks/useAuth.ts, pages/LoginPage.tsx, pages/RegisterPage.tsx, layouts/AuthLayout.tsx

### T5-1 验证路由守卫和布局渲染
- **类型**：test | **依赖**：P0-5
- **测试文件**：`frontend/src/layouts/__tests__/MainLayout.test.tsx`

### P0-6 实现主布局、路由和认证守卫
- **类型**：code | **依赖**：T5-1
- **文件**：src/layouts/MainLayout.tsx, src/components/common/AuthGuard.tsx

### P0-11 实现通用组件（空态/加载/错误边界）
- **类型**：code | **依赖**：P0-6
- **文件**：src/components/common/LoadingSpinner.tsx, ErrorBoundary.tsx, EmptyState.tsx

### T6-1 为 conversationStore 编写单元测试
- **类型**：test | **依赖**：P0-4
- **测试文件**：`frontend/src/store/__tests__/conversationStore.test.ts`

### P0-7 实现 conversationStore
- **类型**：code | **依赖**：T6-1
- **文件**：src/store/conversationStore.ts, src/hooks/useConversation.ts

### T7-1 为 MessageBubble 组件编写测试
- **类型**：test | **依赖**：P0-7
- **测试文件**：`frontend/src/components/chat/__tests__/MessageBubble.test.tsx`

### T7-2 为 ChatInput 组件编写测试
- **类型**：test | **依赖**：P0-7
- **测试文件**：`frontend/src/components/chat/__tests__/ChatInput.test.tsx`

### P0-8 实现对话界面组件
- **类型**：code | **依赖**：T7-1, T7-2
- **文件**：src/components/chat/*.tsx, src/pages/ChatPage.tsx

### T9-1 为 documentStore 编写单元测试
- **类型**：test | **依赖**：P0-4
- **测试文件**：`frontend/src/store/__tests__/documentStore.test.ts`

### P0-9 实现 documentStore 和知识库管理界面
- **类型**：code | **依赖**：T9-1
- **文件**：src/store/documentStore.ts, src/hooks/useDocuments.ts, src/components/documents/*.tsx, src/pages/DocumentsPage.tsx

### T9-2 为 UploadPanel 组件编写测试
- **类型**：test | **依赖**：P0-9
- **测试文件**：`frontend/src/components/documents/__tests__/UploadPanel.test.tsx`

### P0-10 补充 UploadPanel 格式校验与错误处理
- **类型**：code | **依赖**：T9-2
- **文件**：src/components/documents/UploadPanel.tsx

---

## P1 任务（增强性，P0 完成后执行）

### T8-1 为 exportConversation 工具函数编写测试
- **类型**：test | **依赖**：P0-7

### P1-1 实现对话导出工具函数
- **类型**：code | **依赖**：T8-1

### T10-1 为 metricsStore 编写单元测试
- **类型**：test | **依赖**：P0-4

### P1-2 实现 metricsStore 和监控界面
- **类型**：code | **依赖**：T10-1
- **文件**：src/store/metricsStore.ts, MetricsChart.tsx, TrainingStatusCard.tsx, TrainingPage.tsx, MetricsPage.tsx

### P1-3 实现 Docker 部署配置
- **类型**：code | **依赖**：P0-1
- **文件**：frontend/Dockerfile, frontend/nginx.conf, docker-compose.yml
