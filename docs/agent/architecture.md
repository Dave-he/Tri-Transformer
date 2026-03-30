# 系统架构

## 整体架构

```
┌─────────────────────────────────────────────┐
│              浏览器客户端                    │
│         React 18 + Vite + TypeScript        │
│   Ant Design │ Recharts │ Zustand │ Axios   │
└────────────────┬────────────────────────────┘
                 │  HTTP REST + WebSocket
                 ▼
┌─────────────────────────────────────────────┐
│           FastAPI 后端 (Python 3.10)         │
│                                             │
│  ┌─────────────┐    ┌─────────────────────┐ │
│  │  REST API   │    │  WebSocket Handler  │ │
│  │  /api/v1/*  │    │  /ws/detection      │ │
│  └──────┬──────┘    └──────────┬──────────┘ │
│         │                      │            │
│         ▼                      ▼            │
│  ┌─────────────────────────────────────┐    │
│  │         Services Layer              │    │
│  │  HallucinationService │ ModelService│    │
│  └──────────────┬──────────────────────┘    │
│                 │                           │
│                 ▼                           │
│  ┌─────────────────────────────────────┐    │
│  │      Tri-Transformer Model          │    │
│  │         (PyTorch)                   │    │
│  │  Stage1 → Stage2 → Stage3 → Output  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                 │
                 ▼ Docker
┌─────────────────────────────────────────────┐
│            docker-compose                   │
│  frontend:5173  │  backend:8000             │
└─────────────────────────────────────────────┘
```

## 核心模块

### 前端模块
- **pages/**: 检测页面、结果展示页、历史记录页
- **components/**: 文本输入、结果卡片、概率图表、进度指示器
- **hooks/**: useDetection（检测逻辑）、useWebSocket（实时通信）
- **store/**: detectionStore（Zustand）
- **api/**: detectionApi、wsClient

### 后端模块
- **app/api/**: HTTP 路由（/detection、/history、/health）
- **app/services/**: HallucinationDetectionService、ModelService
- **app/model/**: TriTransformerModel（PyTorch 推理）
- **app/schemas/**: DetectionRequest、DetectionResponse
- **app/core/**: 配置、依赖注入、异常处理

## 数据流

```
用户输入文本
  → POST /api/v1/detect
  → HallucinationDetectionService.detect()
  → TriTransformerModel.inference()
  → 三阶段处理（编码 → 对齐 → 聚合）
  → DetectionResponse（分数、标签、置信度）
  → 前端渲染 Recharts 可视化
```

## WebSocket 实时流

```
前端建立 WS 连接 → /ws/detection
  → 发送检测请求
  → 后端流式返回各阶段结果
  → 前端实时更新进度和中间结果
```
