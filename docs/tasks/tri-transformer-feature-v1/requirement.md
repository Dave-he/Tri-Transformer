# 需求文档 - tri-transformer-feature-v1

## 背景

### 缺口 1：MetricsPage 完全不可用（阻塞）

前端 `api/metrics.ts` 和 `api/training.ts` 调用：
- `GET /api/v1/metrics` → 后端不存在，**404**
- `GET /api/v1/training/status` → 后端不存在，**404**

MetricsPage 打开即报错，所有指标卡片和趋势图均无数据。

### 缺口 2：TrainingPage 路径全不匹配（阻塞）

| 前端调用路径 | 后端实际路径 | 状态 |
|------------|------------|------|
| `POST /api/v1/training/start` | `POST /api/v1/train/jobs` | 404 |
| `GET /api/v1/training/progress` | 不存在 | 404 |
| `GET /api/v1/models/available` | 不存在 | 404 |

TrainingPage 无法启动训练、无法查看进度、无法加载模型选项。

### 缺口 3：Chat 消息字段不对齐

后端 `MessageResponse` 返回 `message_id`，前端 `Message` 类型期望 `id`，导致消息 ID 为 undefined。

后端已实现幻觉检测（`hallucination_detected: bool`），但前端 `MessageBubble` 完全未展示该信息，核心功能对用户不可见。

## 验收标准

- AC1: `GET /api/v1/metrics` 正常返回指标数据
- AC2: `GET /api/v1/training/status` 正常返回训练状态
- AC3: `POST /api/v1/train/jobs/start` 创建训练任务返回 `{jobId}`
- AC4: `GET /api/v1/train/jobs/progress` 返回最新任务进度
- AC5: `GET /api/v1/train/jobs/models` 返回可用模型列表
- AC6: 发送消息后 `message.id` 字段有效（不为 undefined）
- AC7: 幻觉检测为 true 时 MessageBubble 显示橙色警告标签
- AC8: 前端 trainingConfig API 路径更新后测试通过
- AC9: 后端/前端全量测试通过
- AC10: flake8 增量无错误
