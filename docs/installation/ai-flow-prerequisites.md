# AI Flow 前置依赖安装指南

> 生成时间：2026-03-30
> 检查模式：full
> 项目类型：全栈项目（Python Backend + React Frontend）

## 📊 检查结果摘要

| 检查项 | 状态 | 版本 | 优先级 | 说明 |
|--------|------|------|--------|------|
| Node.js | ✅ PASS | v22.22.1 | P0 | 必备 |
| Git | ✅ PASS | 2.34.1 | P0 | 必备 |
| 包管理器 | ✅ PASS | pnpm 10.32.1 | P0 | 必备 |
| AI Flow 配置 | ✅ PASS | - | P0 | .ai-flow.config.js 已生成 |
| rd-workflow | ✅ PASS | 0.3.59 | P0 | 最新版本 |
| Figma MCP | ⚠️ 未安装 | - | P1 | 天机MCP 可降级 |
| 天机 MCP | ✅ PASS | - | P1 | 已安装，可作为 Figma 降级方案 |
| playwright MCP | ✅ PASS | - | P1 | 已安装 |
| github MCP | ✅ PASS | - | P1 | 已安装 |
| ESLint | ⚠️ 未配置 | - | P1 | 建议前端项目配置 |

## ⚠️ 阻塞问题

无阻塞问题，可以继续使用 AI Flow。

## 💡 建议安装（推荐）

### Figma MCP（可选）

当前已有天机 MCP 作为降级方案。如需最精准的 Figma 设计稿解析，可安装 Figma MCP：

```bash
# 配置 Figma MCP（需要 Figma Personal Access Token）
flickcli mcp add figma
```

## 🚀 下一步

1. 无阻塞问题，可直接使用 AI Flow
2. 如有 Figma 设计稿需求，建议配置 Figma MCP
3. 使用 AI Flow：直接在对话中说"完成需求 T123456"

---

## 验证安装

重新运行前置检查：

```bash
flickcli skill ai-flow-preflight-check
```
