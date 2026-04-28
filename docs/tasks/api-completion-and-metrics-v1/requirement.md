# 需求文档 - api-completion-and-metrics-v1

## 问题概述

Model API 75%缺失（仅 `/inference` 存在），前后端API路由/字段/响应格式严重不对齐（10处），BGEReranker死代码未接入，无SSE流式端点，Chat DELETE缺失，文档向量化状态轮询缺失。

## 需求清单

| ID | 优先级 | 描述 |
|----|--------|------|
| R1 | P0 | 补全 Model API: GET /status, POST /load, GET /info |
| R2 | P0 | 补全 Chat API: DELETE /sessions/{id} |
| R3 | P1 | 补全 Knowledge API: GET /documents/{id}/status |
| R4 | P0 | 修复前后端API路由/字段/响应格式不对齐(10处) |
| R5 | P1 | 接入 BGEReranker 到搜索管道 |
| R6 | P1 | SSE流式输出端点 |
| R7 | P2 | Train预设配置端点 GET /configs |
| R8 | P0 | 前端重复API模块清理 + 类型对齐 |

## 前后端不对齐详细清单

| # | 前端调用 | 后端实际 | 修复方式 |
|---|---------|---------|---------|
| 1 | POST /documents/upload | POST /knowledge/documents | 统一路由 |
| 2 | POST /documents/search + body | GET /knowledge/search + query | 改后端为POST |
| 3 | 响应 {documents:[]} | 响应 list[Doc] | 包装响应 |
| 4 | 字段 id/name/type/size | 字段 document_id/filename/-/- | 统一字段名 |
| 5 | DELETE 返回 {message} | DELETE 返回 204 空 | 返回 {message} |
| 6 | 消息响应 {message:{}} | 消息响应 MessageResponse 直接 | 包装为 {message:{}} |
| 7 | Conversation 缺 status/message_count | 后端 ConversationItem 含这两个 | 补前端类型 |
| 8 | metrics.ts vs training.ts 重复 | 两个文件完全相同 | 合并为一个 |
| 9 | /model/inference 命名 | PRD 要求 /model/infer | 保持 /inference 不改(向后兼容) |
| 10 | WebSocket /model/stream/stream | PRD 要求 /stream/ws/{id} | 保持现有路径不改 |
