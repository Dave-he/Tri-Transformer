# 技术方案 - tri-transformer-integration-v1

## 方案概述

修复两处阻塞级集成缺口：前后端对话 API 路径对齐 + WebRTC 信令后端实现。

## 变更清单

### 前端修改

**`frontend/src/api/conversations.ts`**
- `GET /conversations` → `GET /chat/sessions`
- `POST /conversations` → `POST /chat/sessions`
- `GET /conversations/:id/messages` → `GET /chat/sessions/:id/history`
- `POST /conversations/:id/messages` → `POST /chat/sessions/:id/messages`

**`frontend/src/mocks/handlers/conversations.ts`**
- 所有 URL 从 `http://localhost:8000` 改为 `http://localhost:8002`
- 路径改为 `/api/v1/chat/sessions*`

**`frontend/src/mocks/handlers/webrtc.ts`**
- URL 从 `http://localhost:8000` 改为 `http://localhost:8002`
- 路径改为 `/api/v1/webrtc/*`

### 后端新增

**`backend/app/api/v1/webrtc.py`**
```
POST /offer      → {sdp: str, type: "answer"}
POST /candidate  → {ok: true}
POST /interrupt  → {ok: true}
```
全部需要 JWT 认证。

**`backend/app/services/model/webrtc_service.py`**
- `WebRTCService.handle_offer()`: echo SDP answer
- `WebRTCService.handle_candidate()`: 记录 candidate
- `WebRTCService.handle_interrupt()`: 设置中断标志

**`backend/app/main.py`**
- 注册 webrtc router，prefix="/api/v1/webrtc"

**`backend/tests/test_webrtc.py`**
- 覆盖 3 个端点的正常流程和 401 场景

## 设计决策

- **不引入 aiortc**：本期目标仅信令层可用，避免重型依赖
- **修改前端而非后端路由**：后端已有 185 个测试，改路径代价更高
