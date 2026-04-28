# 需求文档：修复失败测试 + 补全缺失API + WebSocket实时通信

## 业务目标

- **目标用户**：开发团队
- **要解决的问题**：3个pytest测试持续失败阻塞CI；ChatService缺少会话列表API；前端Chat轮询延迟高
- **成功指标**：pytest 0 failures；API响应<200ms；WebSocket延迟<500ms

## 范围

| 包含 | 不包含 |
|------|--------|
| 修复3个失败测试 | 新增Chat功能特性 |
| 补全GET /api/chat/conversations | 修改已有通过的测试 |
| 前端WebSocket实时通信 | 重构现有API结构 |

## 验收标准（Given/When/Then）

### R1：修复DoraAdapter ImportError (P0)
- **Given** pytest测试环境已配置
- **When** 运行pytest tests/
- **Then** 所有测试通过，DoraAdapter相关import正确解析，0个ImportError

### R2：修复test_galore_config断言失败 (P0)
- **Given** 配置文件包含galore_config相关键值
- **When** 运行pytest tests/test_galore_config.py
- **Then** 所有断言通过，galore_config配置项值与测试期望一致

### R3：补全GET /api/chat/conversations端点 (P0)
- **Given** 用户已认证且存在会话数据 | **When** GET /api/chat/conversations?page=1&page_size=20 | **Then** 返回200，body含conversations数组和pagination元数据
- **Given** 用户已认证但无会话数据 | **When** GET /api/chat/conversations | **Then** 返回200，body含空数组[]
- **Given** 用户已认证 | **When** GET /api/chat/conversations?status=active | **Then** 返回200，只含active会话

### R4：前端WebSocket实时通信 (P1)
- **Given** 用户打开Chat页面且WebSocket可用 | **When** 连接建立成功 | **Then** 延迟<500ms，UI显示connected
- **Given** WebSocket断连 | **When** 自动重连 | **Then** 最多3次间隔5s，UI显示reconnecting
- **Given** 重连失败 | **When** 降级为轮询 | **Then** UI显示offline，轮询间隔5s

## API契约

```
GET /api/chat/conversations
Auth: JWT required
Query: page(int, default=1), page_size(int, default=20, max=100), status(active|archived|all)
Response 200: { conversations: [{id, title, status, created_at, updated_at, message_count}], pagination: {page, page_size, total, total_pages} }
Response 401: { error: "Unauthorized" }
```

## 隐含需求

- WebSocket需要心跳机制防止僵尸连接
- 会话列表按updated_at降序排序
- 分页超出范围返回空列表而非错误
