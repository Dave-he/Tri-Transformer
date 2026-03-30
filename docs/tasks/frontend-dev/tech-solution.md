# 技术方案：Tri-Transformer 前端测试修复

## 背景与目标

前端代码架构完整（6 页面 + 18 组件 + 6 stores + 7 API 文件），TypeScript 零类型错误，但存在 4 个失败测试需修复，阻塞 CI。

目标：**102/102 测试通过，lint/typecheck 零错误。**

## 根因分析

### 问题 1：UploadPanel - Found multiple elements with role="progressbar"

**根因**：Ant Design `Upload.Dragger` 组件内部会渲染带有 `role="progressbar"` 属性的进度条元素。当同时存在自定义 `<div role="progressbar">` 时，`getByRole('progressbar')` 找到多个元素并抛出异常。

**修复方案**：给自定义 progressbar 添加 `data-testid="upload-progress"`，测试改用 `getByTestId('upload-progress')` 替代 `getByRole('progressbar')`。

### 问题 2：UploadPanel - Test timed out

**根因**：`vi.mock` 在文件顶部声明 (hoisted)，但测试内部使用 `await import('../UploadPanel')` 动态导入。在 Vitest 中，动态 import 与 hoisted mock 的交互存在模块缓存问题，导致第二个测试复用了第一个测试的模块缓存，mock 的 `uploadProgress` 值未被正确更新，造成组件等待中的异步 state 永不触发，测试超时。

**修复方案**：在 `beforeEach` 中调用 `vi.resetModules()` 清除模块缓存，确保每个测试独立获取新的 mock 实例。

### 问题 3：AuthGuard - Found multiple elements with text "Protected Content"

**根因**：两个测试用例都使用 `await import('../MainLayout')` 动态导入同一模块，但由于模块缓存，两次 import 返回相同实例。第一个测试的 DOM 未被完全清理（或 React 状态残留），导致第二个测试渲染时同时存在两个 `Protected Content`。

**修复方案**：添加 `beforeEach(() => { vi.resetModules(); })` 确保每次测试用新实例。同时将 mock 的 `useAuthStore` 改为静态 import（非动态），避免缓存干扰。

### 问题 4：AuthGuard - Test timed out

**根因**：同问题 3，模块缓存导致 mock 未正确应用，组件进入等待真实状态逻辑的死循环。

## 文件变更

| 文件 | 变更类型 | 变更内容 |
|------|----------|----------|
| `frontend/src/components/documents/UploadPanel.tsx` | modify | 给自定义 progressbar div 添加 `data-testid="upload-progress"` |
| `frontend/src/components/documents/__tests__/UploadPanel.test.tsx` | modify | 添加 `beforeEach(vi.resetModules)`；`getByRole` 改 `getByTestId` |
| `frontend/src/layouts/__tests__/MainLayout.test.tsx` | modify | 改为静态 import AuthGuard；添加 `beforeEach(vi.resetModules)` |

## 验证策略

```bash
cd frontend && npx vitest run       # 102/102 通过
cd frontend && npx tsc --noEmit     # 0 错误
cd frontend && npx eslint src/components/documents/__tests__/UploadPanel.test.tsx src/layouts/__tests__/MainLayout.test.tsx src/components/documents/UploadPanel.tsx
```

## 风险与注意事项

- 修复只涉及测试文件和极小的组件属性变更（`data-testid`），不改变任何业务逻辑
- `vi.resetModules()` 会清除所有模块缓存，因此后续的 `await import()` 必须放在 `it` 块内部（非顶层）
- Ant Design 内部 DOM 结构可能随版本变化，建议统一使用 `data-testid` 而非依赖 ARIA role 做断言
