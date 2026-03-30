# 任务清单 — Tri-Transformer 后端增量开发 v2

**任务 ID**: tri-transformer-backend-v2
**来源**: tech-solution.yaml
**生成时间**: 2026-03-27

---

## 概述

4 个功能模块，8 个任务（4 test + 4 code），TDD 顺序执行。

**执行顺序**: T1-1 → T1-2 → P0-1 → T2-1 → P0-2 → T3-1 → P0-3 → T4-1 → P0-4

---

## 测试命令

```bash
cd backend && pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 任务列表

### [RED] T1-1: LoraAdapter 单元测试
- **文件**: `backend/tests/test_pluggable_llm.py`
- **Given**: nn.Linear(64,64)
- **When**: LoraAdapter 包装
- **Then**: shape 正确 / freeze 生效 / 参数量 < 5%

### [RED] T1-2: PluggableLLMAdapter 单元测试
- **文件**: `backend/tests/test_pluggable_llm.py`（追加）
- **Given**: Mock TransformerEncoder(d_model=64)
- **Then**: forward 正确 / LoRA 注入 / 梯度可反传

### [GREEN] P0-1: 实现 LoraAdapter + PluggableLLMAdapter
- **文件**:
  - `backend/app/model/lora_adapter.py`
  - `backend/app/model/pluggable_llm.py`
- **DoD**: T1-1 + T1-2 全部通过

---

### [RED] T2-1: 多模态 Tokenizer 单元测试
- **文件**: `backend/tests/test_tokenizer.py`
- **Given**: 文本/音频帧数组/图像 bytes
- **Then**: ID 区间正确 / 不越界 / 特殊 Token 可查

### [GREEN] P0-2: 实现统一 Tokenizer
- **文件**:
  - `backend/app/model/tokenizer/__init__.py`
  - `backend/app/model/tokenizer/text_tokenizer.py`
  - `backend/app/model/tokenizer/audio_tokenizer.py`
  - `backend/app/model/tokenizer/vision_tokenizer.py`
  - `backend/app/model/tokenizer/unified_tokenizer.py`
- **DoD**: T2-1 全部通过

---

### [RED] T3-1: WebSocket 流式推理测试
- **文件**: `backend/tests/test_stream.py`
- **Given**: JWT token，TestClient WebSocket
- **Then**: 连接/鉴权/流式 token/interrupt/关闭

### [GREEN] P0-3: 实现 StreamingEngine + WebSocket 路由
- **文件**:
  - `backend/app/services/model/stream_engine.py`
  - `backend/app/api/v1/stream.py`
  - `backend/app/main.py`（修改）
- **DoD**: T3-1 全部通过

---

### [RED] T4-1: FactChecker 和幻觉阻断集成测试
- **文件**: `backend/tests/test_fact_checker.py`
- **Given**: FactChecker Mock embedding，对话会话
- **Then**: score/hallucination_detected/对话响应字段

### [GREEN] P0-4: 实现 FactChecker + 对话链路集成
- **文件**:
  - `backend/app/services/model/fact_checker.py`
  - `backend/app/schemas/chat.py`（修改）
  - `backend/app/services/chat/chat_service.py`（修改）
  - `backend/app/models/chat_session.py`（修改）
- **DoD**: T4-1 全部通过 + 已有 80 个测试不破坏

---

## 验收标准

- [ ] pytest tests/ 全部通过（>=100 个）
- [ ] 覆盖率 >= 80%
- [ ] LoRA 参数量 < 底座 5%
- [ ] 各模态 Token ID 区间互不重叠
- [ ] WebSocket 流式响应正常
- [ ] 对话响应含 hallucination_detected 字段
