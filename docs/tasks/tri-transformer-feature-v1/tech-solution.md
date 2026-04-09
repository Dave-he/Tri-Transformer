# 技术方案 - tri-transformer-feature-v1

## 后端新增

### metrics.py
- `GET /metrics`：聚合 completed TrainJob 的 `config["metrics"]` 历史，生成 `{current, history}`
- `GET /training/status`：查最新 running/pending job，无则返回 idle

### train.py 新增路由
- `POST /jobs/start`：与 `POST /jobs` 同逻辑，返回 `{jobId}`（前端期望 camelCase）
- `GET /jobs/progress`：查最新运行中 job 进度
- `GET /jobs/models`：静态 4 个预设模型列表

### main.py
- 注册 metrics router，prefix="/api/v1/metrics"

## 前端修改

### conversations.ts
字段映射：`message_id` → `id`，`hallucination_detected` → `hallucinationDetected`，`created_at` → `createdAt`

### trainingConfig.ts
路径全部对齐后端真实路径（`/train/jobs/*`）

### types/api.ts
`Message.hallucinationDetected?: boolean`

### MessageBubble.tsx
助手消息 `hallucinationDetected === true` → Ant Design `<Tag color="orange">⚠ 检测到幻觉</Tag>`
