# 系统架构

## 整体数据流

```
浏览器 (React 18)
  │  HTTP REST + WebSocket/SSE
  ▼
FastAPI 后端 (Python 3.10)
  ├── API Layer: /api/v1/{auth,chat,knowledge,model,train,stream}
  ├── Services: ChatService, InferenceService, FactChecker, RAGRetriever
  ├── Model: TriTransformerModel (PyTorch)
  │     ITransformer  ← Qwen3-8B/14B Dense 骨干（左端插拔）
  │       └── Qwen3DecoderLayer × 36（GQA 32/8 + QK-Norm + RoPE θ=1M）
  │     CTransformer  ← DiT 控制中枢（维度对齐 Qwen3 hidden=4096）
  │       └── State Slots × 16 + Cross-Attn + adaLN-Zero
  │     OTransformer  ← Qwen3-8B/32B Dense 骨干（右端插拔）
  │       └── Planning Encoder + Streaming Decoder
  │     Thinking Mode ← enable_thinking=True/False 动态切换
  └── DB: SQLAlchemy async (SQLite dev / PostgreSQL prod)
          Milvus (vector store，替代 ChromaDB 用于生产)
          ChromaDB (vector store，开发/测试)
```

## Qwen3 骨干配置（推荐规格）

| 位置 | 推荐模型 | 参数量 | 关键超参 | 显存占用 |
|---|---|---|---|---|
| I-Transformer（左端）| Qwen3-8B | 8.2B | 36层, hidden=4096, GQA 32/8, rope_θ=1M | ~16GB BF16 |
| C-Transformer（控制层）| 自研 DiT | ~0.5B | 8层, d=4096, 兼容 Qwen3 维度 | ~1GB |
| O-Transformer（右端）| Qwen3-8B | 8.2B | 同上 | ~16GB BF16 |
| MoE 方案（单卡全系统）| Qwen3-30B-A3B | 30B总/3B激活 | 48层, 128专家取8 | ~60GB 加载/~6GB推理 |

**Thinking Mode 策略**：
- 实时对话（< 300ms 延迟）→ `enable_thinking=False`（Non-Thinking Mode）
- 复杂知识推理 → `enable_thinking=True`（触发内部 CoT `<think>...</think>`）

## 前端架构

**Feature-based modular with Zustand state management**

```
App.tsx (BrowserRouter + Ant Design ConfigProvider)
  └── Layouts (MainLayout / AuthLayout)
        └── Pages (Chat, Documents, Training, Metrics, Login, Register)
              ├── Components (chat/, common/, documents/, metrics/, training/)
              ├── Hooks (useConversation, useAuth, ...)
              └── Store (Zustand) ← API clients (axios)
```

关键约定：
- `@/` path alias 指向 `frontend/src/`
- 组件消费 Zustand store，不直接调用 API
- Optimistic updates: UI 先更新，API 后确认
- MSW (`mocks/handlers/`) 在测试中拦截 API

## 后端架构

**Layered + Dependency Injection**

```
Routes (app/api/v1/)
  └── Depends() injection: DB session, current user
        └── Services (app/services/)
              ├── chat/: ChatService, RAGRetriever, Reranker
              ├── model/: InferenceService, FactChecker
              └── train/: TrainingService
                    └── Model (app/model/)
                          ├── tri_transformer.py  # 主模型
                          ├── branches.py         # I/C/O 三分支
                          ├── lora_adapter.py     # LoRA fine-tuning
                          └── pluggable_llm.py    # 可插拔 LLM 接口
```

关键约定：
- 全异步 (async/await)，DB 操作使用 `async with db.begin()`
- Pydantic v2 schemas 分离于 SQLAlchemy models
- `app/dependencies.py` 集中管理所有 Depends 工厂

## Eval Pipeline

```
eval/
  ├── ground_truth/        # Ground truth 生成
  ├── loss/                # 自定义损失函数
  │     ├── hallucination_loss.py
  │     ├── rag_loss.py
  │     ├── control_alignment_loss.py
  │     └── total_loss.py
  ├── pipeline/            # 评估管道
  │     ├── hallucination_evaluator.py
  │     ├── rag_evaluator.py
  │     ├── dialog_evaluator.py
  │     └── ci_gate.py     # CI 门禁
  └── scripts/             # CLI 脚本
```

CI 门禁通过 `.github/workflows/eval-ci.yml` 自动触发。

## API 端点概览

| Method | Path | 说明 | Auth |
|--------|------|------|------|
| POST | `/api/v1/auth/register` | 注册 | No |
| POST | `/api/v1/auth/login` | 登录(JWT) | No |
| GET | `/health` | 健康检查 | No |
| POST | `/api/v1/chat/sessions` | 创建会话 | Yes |
| POST | `/api/v1/chat/sessions/{id}/messages` | 发送消息 | Yes |
| GET/POST | `/api/v1/knowledge` | 文档管理 | Yes |
| GET | `/api/v1/model` | 模型信息 | Yes |
| POST | `/api/v1/train` | 创建训练任务 | Yes |
| WS | `/api/v1/model/stream` | 流式输出 | Yes |

## Jetson Nano 边缘部署架构

**Jetson Nano 8GB (Maxwell GPU, CUDA 10.2, aarch64)**

```
[Jetson Nano 8GB]
    ├── FastAPI Backend (CPU)
    │     ├── InferenceService (pytorch_direct)
    │     │     └── TriTransformerModel.forward() → 三分支全功能推理
    │     │     ├── 幻觉检测: C-Transformer 实时监控
    │     │     ├── RAG 检索: ChromaDB + BM25 + BGE Reranker
    │     │     └── 实时打断: I/O KV-Cache 动态截断
    │     ├── LlamaCppService (llamacpp_gguf)
    │     │     └── llama-cpp-python Llama(Q5_K_M.gguf) → 单分支轻量推理
    │     │     └── 仅 O-Transformer Streaming Decoder (无幻觉检测/控制)
    │     ├── 推理模式切换: inference_mode 配置
    │     └── RAG pipeline (ChromaDB + sentence-transformers)
    ├── Training (GPU + CPU, 共享 8GB 内存)
    │     ├── JetsonDevice.detect() → 硬件适配
    │     ├── TriTransformerTrainer(GaLore+AMP+grad_accum)
    │     ├── MemoryMonitor → 85%阈值 WARNING
    │     └── 三阶段: LoRA(I+O) → LoRA(C) → 全量(可选)
    └── CLI Scripts
        ├── train.py --jetson-nano
        ├── convert_to_gguf.py
        ├── install_jetson_deps.sh
        └── build_llamacpp_jetson.sh
```

**推理模式对比**：

| 模式 | 架构 | 内存占用 | 功能 | 适用场景 |
|------|------|---------|------|---------|
| pytorch_direct | I/C/O 三分支 | ~619MB FP16 | 全功能（幻觉检测+RAG+打断） | 默认推荐 |
| llamacpp_gguf | O 单分支 | ~1.8GB Q5_K_M | 仅文本生成（无幻觉检测） | 轻量部署 |

**关键硬件限制**：

| 约束 | 影响 | 缓解 |
|------|------|------|
| CUDA 10.2 only | PyTorch <=1.13.x, 无 torch.compile | JetPack PyTorch wheel |
| Maxwell sm_53 | 无 FlashAttention/DeepSpeed/vLLM | 标准注意力 + GaLore |
| 8GB 共享内存 | CPU+GPU 共用 | GaLore(rank=64) + AMP + 85%监控 |
| 无 Tensor Cores | AMP 仅节省内存无速度提升 | grad_accum=4 补偿 |
