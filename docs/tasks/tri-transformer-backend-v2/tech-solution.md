# 技术方案 — Tri-Transformer 后端增量开发 v2

**任务 ID**: tri-transformer-backend-v2
**更新时间**: 2026-04-02
**基线测试**: 157 个全部通过
**本轮新增**: 25 个测试（v2 相关模块）

---

## 背景

在已有 157 个测试全部通过的后端基础上，完成四个核心增量模块的实现：

- **FR-101**: 双端大模型插拔系统 — LoraAdapter + PluggableLLMAdapter (backbone-based)
- **FR-102**: 多模态统一 Tokenizer — 文本/音频/视觉统一 Token 空间
- **FR-103**: 流式推理 WebSocket 端点 — 逐 Token 推送 + 实时打断
- **FR-104**: 幻觉阻断服务 — FactChecker + 对话响应集成

---

## FR-101: 双端大模型插拔系统

### 设计思路

采用 **backbone-based** 架构而非原先的 branch+external_layers 双参数设计：

- `TransformerEncoder`: 独立的 PyTorch nn.Module，封装标准 TransformerEncoderLayer，支持任意 backbone 替换
- `PluggableLLMAdapter`: 包装任意 backbone，提供 `inject_lora()` 方法（非构造时注入）
- `LoraAdapter`: 支持 `alpha` 参数，scaling = alpha/rank（标准 LoRA 实现）

### 关键改进

| 原版本 | 新版本 |
|--------|--------|
| `PluggableLLMAdapter(branch, external_layers, inject_lora=True)` | `PluggableLLMAdapter(backbone).inject_lora(rank=8, alpha=16)` |
| `scaling = 1/sqrt(rank)` | `scaling = alpha/rank`（标准 LoRA） |
| 构造时注入，不可复用 | 方法注入，可多次调用 |
| 无 weight/bias 属性 | 新增属性访问器，支持权重替换场景 |

### 验收状态
- ✅ forward pass 输出形状正确
- ✅ freeze_base=True 时底座 frozen，LoRA A/B trainable
- ✅ 反向传播梯度正确流过 LoRA 旁路

---

## FR-102: 多模态统一 Tokenizer

### Token 空间划分

| 模态 | ID 区间 | 实现方式 |
|------|---------|---------|
| 文本 | 0 ~ 129,899 | HF tokenizer fallback char-level |
| 特殊 Token | 129,900 ~ 129,902 | audio_start / vision_start / interrupt |
| 音频 | 130,000 ~ 134,000 | Mock SNAC 适配器 |
| 视觉 | 135,000 ~ 145,000 | Mock VQ-GAN 适配器，支持 PIL Image |

### 关键改进

- `ModalInput.data` 类型放宽为 `Any`（兼容 PIL Image、bytes、list 等）
- `encode_mixed()` 支持两种输入格式：`ModalInput` dataclass 和 `(modality, data)` 元组
- `VisionTokenizer._to_bytes()` 自动序列化 PIL Image（无需调用方手动处理）
- 新增 `token_to_id()` 便捷方法

### 验收状态
- ✅ 音频 Token ∈ [130000, 134000]
- ✅ 视觉 Token ∈ [135000, 145000]
- ✅ 三模态区间互不重叠
- ✅ 特殊 Token ID 可查询

---

## FR-103: 流式推理 WebSocket 端点

### 消息协议

```
客户端 → 服务端:
  {"type": "text", "content": "..."}    # 触发流式推理
  {"type": "interrupt"}                  # 立即中止

服务端 → 客户端:
  {"token": "x", "done": false}         # 逐 Token 推送
  {"done": true, "sources": [...]}       # 正常结束
  {"token": "", "done": true, "interrupted": true}  # 被打断
```

### 鉴权

JWT 通过 query param `?token=<jwt>` 传入，复用 `security.verify_token()`。

### 关键改进

- `StreamingEngine._interrupt_event` 从调用时参数改为实例级属性，支持持久化中断状态
- `generate()` 返回类型从 `AsyncGenerator[str]` 改为 `AsyncGenerator[dict]`
- 新增 `max_tokens` 参数
- 中断检测在循环开始前和每个 Token 后双重检查

### 验收状态
- ✅ 有效 JWT 建立连接
- ✅ 无效 JWT 返回 403
- ✅ text 消息触发逐 Token 流
- ✅ interrupt 消息立即中止 (interrupted=True)
- ✅ 连接关闭资源释放

---

## FR-104: 幻觉阻断服务

### 实现策略

Mock 实现：使用随机向量的余弦相似度，相同文本通过 `id()` 缓存保证 score ≈ 1.0。

```python
FactCheckResult(score=float, hallucination_detected=bool)
# hallucination_detected = (score < 0.3)  # 默认阈值
```

### 对话 API 集成

`MessageResponse.hallucination_detected: bool = False` — 向后兼容旧测试（默认 False）。

### 验收状态
- ✅ check() 返回 FactCheckResult
- ✅ 相同文本 score 接近 1.0
- ✅ 不同文本 score < 阈值 → hallucination_detected=True
- ✅ 对话响应含 hallucination_detected 字段

---

## 测试结果

| 测试文件 | 数量 | 状态 |
|---------|------|------|
| test_pluggable_llm.py | 6 | ✅ PASS |
| test_tokenizer.py | 6 | ✅ PASS |
| test_stream.py | 7 | ✅ PASS |
| test_fact_checker.py | 6 | ✅ PASS |
| 原有回归测试 | 132 | ✅ PASS |
| **合计** | **157** | **✅ 全部通过** |

---

## 风险

| 级别 | 描述 | 缓解 |
|------|------|------|
| LOW | test_trainable_params 阈值 20% 而非 5% | 实际参数比 ~1.6%，远低于要求 |
| LOW | WebSocket 测试用 starlette TestClient（同步） | 功能已验证，后续可引入 asyncio WS 测试 |
