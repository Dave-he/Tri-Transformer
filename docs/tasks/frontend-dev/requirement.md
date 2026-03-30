# 需求文档：Tri-Transformer 前端工程开发

## 概述

推进 Tri-Transformer 前端工程开发，基于现有 React + TypeScript + Vite 代码库，修复存在的测试失败、完善代码质量，确保整个前端工程可正常运行、测试全绿、lint 零错误。

## 业务目标

为 Tri-Transformer AI 系统提供完整可用的 Web 管理界面，支持：
- 实时多模态对话（文本 + WebRTC 音视频）
- RAG 知识库管理（文档上传、检索测试）
- 训练监控（训练状态、超参配置）
- 性能指标展示（检索准确率、BLEU、幻觉率）

## 现有代码状态

| 类别 | 状态 | 详情 |
|------|------|------|
| 页面 | ✅ 完整 | 6 个页面（Login/Register/Chat/Documents/Training/Metrics） |
| 组件 | ✅ 完整 | 18 个组件 |
| Store | ✅ 完整 | 6 个 Zustand stores |
| API 层 | ✅ 完整 | 7 个 API 文件 |
| TypeScript | ✅ 零错误 | `tsc --noEmit` 通过 |
| 测试 | ⚠️ 部分失败 | 102 个测试，4 个失败 |

## 需修复问题

### 问题 1：UploadPanel 测试失败
- `shows progress bar when uploadProgress is set` → Found multiple elements with role="progressbar"
- `renders upload area` → Test timed out in 5000ms

### 问题 2：AuthGuard 测试失败
- `renders children when authenticated` → Found multiple elements with text "Protected Content"
- `does not render protected content when not authenticated` → Test timed out in 5000ms

## 验收标准

| ID | 标准 | 可测验 |
|----|------|--------|
| AC-01 | 所有 102 个 Vitest 测试通过 | `pnpm test` |
| AC-02 | TypeScript 零类型错误 | `pnpm typecheck` |
| AC-03 | ESLint 零 warning/error | `pnpm lint` |
| AC-04 | UploadPanel 进度条测试修复 | 包含在 AC-01 |
| AC-05 | AuthGuard 认证测试修复 | 包含在 AC-01 |
| AC-06 | 前端可正常构建 | `pnpm build` |

## 技术栈

- React 18 + TypeScript + Vite
- Ant Design 5.x
- Zustand（状态管理）
- Axios（HTTP 客户端）
- Recharts（图表）
- MSW（Mock Service Worker，测试 mock）
- Vitest + @testing-library/react（测试框架）
