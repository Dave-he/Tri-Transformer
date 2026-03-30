# 任务清单 - Tri-Transformer 前端 v2

## 概述

共 **20 个任务**：**10 个 test 任务** + **10 个 code 任务**

- P0 任务：20 个
- 测试优先：所有 code 任务均依赖对应 test 任务

## 任务列表

### 模块 1：类型定义

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T1-1 | test | P0 | 验证 WebRTC 类型定义完整性 | `src/types/__tests__/api.test.ts` |
| P0-1 | code | P0 | 新增 WebRTC 与训练配置 TypeScript 类型 | `src/types/webrtc.ts`, `src/types/trainingConfig.ts` |

### 模块 2：API 客户端

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T2-1 | test | P0 | WebRTC API 单元测试 | `src/api/__tests__/webrtc.test.ts` |
| P0-2 | code | P0 | WebRTC + 训练配置 API 客户端 | `src/api/webrtc.ts`, `src/api/trainingConfig.ts` |

### 模块 3：MSW Mock

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T3-1 | test | P0 | 验证新 MSW handlers 拦截 | `src/mocks/__tests__/handlers.test.ts` |
| P0-3 | code | P0 | WebRTC + 训练配置 MSW handlers | `src/mocks/handlers/webrtc.ts`, `src/mocks/handlers/trainingConfig.ts`, `src/mocks/server.ts` |

### 模块 4：WebRTC Store

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T4-1 | test | P0 | webrtcStore 状态机测试 | `src/store/__tests__/webrtcStore.test.ts` |
| P0-4 | code | P0 | webrtcStore 实现 | `src/store/webrtcStore.ts` |

### 模块 5：训练配置 Store

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T5-1 | test | P0 | trainingConfigStore 测试 | `src/store/__tests__/trainingConfigStore.test.ts` |
| P0-5 | code | P0 | trainingConfigStore 实现 | `src/store/trainingConfigStore.ts` |

### 模块 6：WebRTC 控制组件

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T6-1 | test | P0 | WebRTCControls 组件测试 | `src/components/chat/__tests__/WebRTCControls.test.tsx` |
| P0-6 | code | P0 | WebRTCControls 实现 | `src/components/chat/WebRTCControls.tsx` |

### 模块 7：音频波形可视化

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T7-1 | test | P0 | AudioVisualizer 组件测试 | `src/components/chat/__tests__/AudioVisualizer.test.tsx` |
| P0-7 | code | P0 | AudioVisualizer 实现 | `src/components/chat/AudioVisualizer.tsx` |

### 模块 8：模态切换与 ChatPage 集成

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T8-1 | test | P0 | ChatModeTabs 组件测试 | `src/components/chat/__tests__/ChatModeTabs.test.tsx` |
| P0-8 | code | P0 | ChatModeTabs + ChatPage 集成 | `src/components/chat/ChatModeTabs.tsx`, `src/pages/ChatPage.tsx` |

### 模块 9：大模型插件选择

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T9-1 | test | P0 | ModelPluginSelector 组件测试 | `src/components/training/__tests__/ModelPluginSelector.test.tsx` |
| P0-9 | code | P0 | ModelPluginSelector 实现 | `src/components/training/ModelPluginSelector.tsx` |

### 模块 10：训练配置表单

| ID | 类型 | 优先级 | 标题 | 文件 |
|----|------|--------|------|------|
| T10-1 | test | P0 | TrainingConfigForm 组件测试 | `src/components/training/__tests__/TrainingConfigForm.test.tsx` |
| P0-10 | code | P0 | TrainingConfigForm + TrainingPage 集成 | `src/components/training/TrainingConfigForm.tsx`, `src/pages/TrainingPage.tsx` |

## 执行顺序（TDD：先 RED 后 GREEN）

### 第一轮（RED 阶段）：写所有测试文件

1. T1-1 → T2-1 → T3-1 → T4-1 → T5-1 → T6-1 → T7-1 → T8-1 → T9-1 → T10-1

### 第二轮（GREEN 阶段）：写实现文件让测试通过

1. P0-1 → P0-2 → P0-3 → P0-4 → P0-5 → P0-6 → P0-7 → P0-8 → P0-9 → P0-10

## 验收命令

```bash
# 运行所有测试（含 v1 54 个 + v2 新增约 30 个）
cd frontend && npx vitest run

# TypeScript 类型检查
cd frontend && npx tsc --noEmit
```

## 完成标准

- 全量 vitest 通过（无回归）
- TypeScript 0 错误
- 8 条 AC（AC-V2-001 ~ AC-V2-008）全部满足
