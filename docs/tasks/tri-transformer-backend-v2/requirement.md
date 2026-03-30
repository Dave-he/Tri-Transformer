# 需求文档 — Tri-Transformer 后端增量开发 v2

**任务 ID**: tri-transformer-backend-v2
**前置任务**: tri-transformer-backend（80 tests passing）
**创建时间**: 2026-03-27

---

## 背景

当前后端服务已完整实现 FastAPI 核心链路（认证/RAG/对话/推理Mock/训练调度），80 个测试全部通过，覆盖率 76%。

最新 sub_prds 01~04 定义了四个尚未实现的核心后端模块，本任务针对性增量交付。

---

## 功能需求

### FR-101：双端大模型插拔系统

**模块**：`backend/app/model/pluggable_llm.py` + `lora_adapter.py`

- `HFModelLoader`：根据 model_id + 层数配置，从 HF Hub 或本地加载权重层
- `PluggableLLMAdapter`：将外部模型权重映射到 ITransformer/OTransformer 分支
- `LoraAdapter`：为任意 `nn.Linear` 注入可训练的低秩旁路（rank-r A/B 矩阵）
- 支持 `freeze_base=True` 冻结底座 + LoRA 旁路训练

**验收标准**：
1. Mock 小模型加载后 forward pass 不报错
2. LoRA 注入后梯度仅流经 A/B 矩阵，底座参数 `requires_grad=False`
3. `num_trainable_params` < `num_base_params` 时 LoRA 生效

---

### FR-102：多模态统一 Tokenizer

**模块**：`backend/app/model/tokenizer/`

- `TextTokenizer`：封装 HF tokenizer（fallback to char-level mock）
- `AudioTokenizer`：将音频帧（float array）→ 离散 Token（范围 [130000, 134000]）
- `VisionTokenizer`：将图像帧（PIL Image）→ 视觉 Token（范围 [135000, 145000]）
- `UnifiedTokenizer`：按模态 flag 合并为单一 `input_ids` 序列，注册特殊 Token

**验收标准**：
1. 混合编码序列中各模态 Token ID 区间互不重叠
2. 特殊 Token `<|audio_start|>` / `<|vision_start|>` / `<|interrupt|>` 可查询 ID
3. 音频 Token ID ∈ [130000, 134000]，视觉 Token ID ∈ [135000, 145000]

---

### FR-103：流式推理 WebSocket 端点

**模块**：`backend/app/api/v1/stream.py` + `backend/app/services/model/stream_engine.py`

- WebSocket 路由 `/api/v1/model/stream`
- `StreamingEngine`：维护 I 侧增量状态，驱动 O 侧逐 Token 自回归生成
- 支持实时打断（`{"type":"interrupt"}`）立即中止当前生成循环
- Mock 模式：直接从预定义字符串逐 Token 推送

**验收标准**：
1. WebSocket 连接/握手/鉴权 通过
2. 收到 text 消息后逐步返回 `{"token":"x","done":false}` 流
3. 收到 interrupt 消息后立即发送 `{"done":true,"interrupted":true}`
4. 连接关闭后资源正确释放

---

### FR-104：幻觉阻断服务

**模块**：`backend/app/services/model/fact_checker.py`

- `FactChecker.check(generated: str, contexts: list[str]) -> FactCheckResult`
- 使用余弦相似度（embedding-based，Mock 可用随机向量）判断一致性
- 一致性 < 0.3 → `hallucination_detected=True`
- 对话 API 响应增加 `hallucination_detected` 字段

**验收标准**：
1. `check()` 返回 `FactCheckResult(score=float, hallucination_detected=bool)`
2. 完全相同文本一致性接近 1.0
3. 对话响应 schema 新增 `hallucination_detected: bool` 字段

---

## 非功能需求

- 新增代码测试覆盖率 >= 80%
- 不破坏已有 80 个测试
- LoRA 参数量 < 底座参数量 5%

---

## 技术栈约束

| 组件 | 库 |
|------|-----|
| HF 权重加载 | `transformers>=4.40` |
| LoRA | `peft>=0.10` 或自实现 |
| WebSocket | FastAPI 原生 WebSocket |
| 事实校验 | `sentence-transformers`（Mock 可用 random）|
| 音频 Codec | Mock 实现（SNAC 真实集成后续） |
| 视觉 Token | Mock 实现（VQ-GAN 后续）|
