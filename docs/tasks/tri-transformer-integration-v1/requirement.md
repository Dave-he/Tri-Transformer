# 需求文档 - tri-transformer-integration-v1

## 背景

项目探索分析发现两处阻塞级集成缺口，导致核心功能无法联调：

### 缺口 1：对话 API 路径不匹配（阻塞）

| 位置 | 实际路径 |
|------|---------|
| 前端 `api/conversations.ts` | `/conversations`, `/conversations/:id/messages` |
| 后端 `api/v1/chat.py` | `/sessions`, `/sessions/:id/messages`, `/sessions/:id/history` |
| 后端挂载前缀 `main.py` | `/api/v1/chat` |

前端期望的完整路径 `/api/v1/conversations` 与后端实际 `/api/v1/chat/sessions` 完全不匹配，对话列表、新建对话、发送消息全部 404。

### 缺口 2：WebRTC 信令后端缺失（阻塞）

前端已有完整实现：
- `webrtcStore.ts`：完整的 RTCPeerConnection 管理
- `api/webrtc.ts`：`sendOffer / sendCandidate / sendInterrupt` 三个 API 调用
- `types/webrtc.ts`：完整类型定义

但后端 `app/api/v1/` 目录下**无 webrtc.py**，三个端点全部 404，音视频通话完全无法启动。

### 缺口 3：MSW mock URL 硬编码错误（次要）

前端 mock handler 硬编码 `http://localhost:8000`，但 `client.ts` baseURL 为 `http://localhost:8002`，
导致测试中 mock 无法正确拦截请求。

## 验收标准

- AC1: `GET /api/v1/chat/sessions` 正确返回会话列表
- AC2: `POST /api/v1/chat/sessions` 创建会话返回 201
- AC3: `GET /api/v1/chat/sessions/{id}/history` 返回消息历史
- AC4: `POST /api/v1/chat/sessions/{id}/messages` 返回 AI 回复
- AC5: `POST /api/v1/webrtc/offer` 返回 `{sdp, type: "answer"}`
- AC6: `POST /api/v1/webrtc/candidate` 返回 `{ok: true}`
- AC7: `POST /api/v1/webrtc/interrupt` 返回 `{ok: true}`
- AC8: 后端 pytest 全部通过，前端 vitest 全部通过
- AC9: flake8 无新增 E/F 错误
