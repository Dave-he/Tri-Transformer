# 任务清单 - tri-transformer-integration-v1

## TDD 执行顺序

### RED 阶段（先写测试）

| ID | 任务 | 文件 |
|----|------|------|
| T1 | [RED] test_webrtc_offer - POST /webrtc/offer 返回 SDP answer | tests/test_webrtc.py |
| T2 | [RED] test_webrtc_candidate - POST /webrtc/candidate 返回 ok | tests/test_webrtc.py |
| T3 | [RED] test_webrtc_interrupt - POST /webrtc/interrupt 返回 ok | tests/test_webrtc.py |
| T4 | [RED] test_webrtc_offer_no_auth - 无 token 返回 401 | tests/test_webrtc.py |

### GREEN 阶段（写代码让测试通过）

| ID | 任务 | 文件 |
|----|------|------|
| T5 | [GREEN] 新建 webrtc_service.py | app/services/model/webrtc_service.py |
| T6 | [GREEN] 新建 webrtc.py 路由 | app/api/v1/webrtc.py |
| T7 | [GREEN] 注册 webrtc router | app/main.py |
| T8 | [GREEN] 对齐前端 conversations.ts 路径 | frontend/src/api/conversations.ts |
| T9 | [GREEN] 更新前端 MSW mock URL | frontend/src/mocks/handlers/*.ts |

### VERIFY 阶段

| ID | 任务 |
|----|------|
| T10 | 后端全量 pytest ≥ 185 passed |
| T11 | flake8 增量检查 0 errors |
