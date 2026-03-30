# 任务清单：Tri-Transformer 前端测试修复

## 概述

修复 4 个失败测试用例，实现 102/102 测试通过。

## 任务依赖关系

```
T1-1 ──┐
T1-2 ──┤── P0-1 ── P0-2 ──┐
T2-1 ──┐                   ├── P0-4（验收）
T2-2 ──┴── P0-3 ───────────┘
```

## 任务清单

### 测试任务（先写 RED 测试）

| ID | 优先级 | 标题 | 文件 |
|----|--------|------|------|
| T1-1 | P0 | 验证 UploadPanel 进度条 data-testid 断言 | UploadPanel.test.tsx |
| T1-2 | P0 | 验证 UploadPanel 渲染无 timeout | UploadPanel.test.tsx |
| T2-1 | P0 | 验证 AuthGuard 认证通过无 multiple elements | MainLayout.test.tsx |
| T2-2 | P0 | 验证 AuthGuard 未认证重定向无 timeout | MainLayout.test.tsx |

### 代码任务（后写 GREEN 实现）

| ID | 优先级 | 标题 | 文件 | 依赖 |
|----|--------|------|------|------|
| P0-1 | P0 | 给 UploadPanel progressbar 添加 data-testid | UploadPanel.tsx | T1-1 |
| P0-2 | P0 | 修复 UploadPanel 测试选择器和模块缓存 | UploadPanel.test.tsx | T1-1, T1-2, P0-1 |
| P0-3 | P0 | 修复 AuthGuard 测试静态 import 和缓存清理 | MainLayout.test.tsx | T2-1, T2-2 |
| P0-4 | P0 | 运行完整测试套件验收 | - | P0-1, P0-2, P0-3 |

## 执行命令

```bash
# 运行测试
cd frontend && npx vitest run

# TypeScript 检查
cd frontend && npx tsc --noEmit

# 增量 lint
cd frontend && npx eslint src/components/documents/UploadPanel.tsx src/components/documents/__tests__/UploadPanel.test.tsx src/layouts/__tests__/MainLayout.test.tsx
```

## 验收标准

- AC-01: 102/102 Vitest 测试通过
- AC-02: TypeScript 零类型错误
- AC-03: ESLint 零 warning/error
