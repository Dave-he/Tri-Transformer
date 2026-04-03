# 任务清单 — Tri-Transformer 后端增量开发 v2

**任务 ID**: tri-transformer-backend-v2
**创建时间**: 2026-04-02
**基线测试**: 157/157 全部通过
**模式**: 补强测试覆盖缺口（代码已实现）

---

## 背景

FR-101~FR-104 的代码实现已全部完成，157 个测试全部通过。
本 plan 聚焦于**补强测试覆盖的三个缺口**：

1. **LoRA 参数比** — 现有测试检查 `< 20%`，需补强为精确 `< 5%`
2. **StreamingEngine done 消息** — 需验证完整流式响应结构
3. **VisionTokenizer PIL Image** — 需测试直接传入 PIL Image 的路径

---

## 任务列表

### P0 测试任务（RED 阶段）

| ID | 任务 | 测试文件 | 新增用例 |
|----|------|---------|---------|
| T0-1 | LoRA 参数量精确比例测试 | test_pluggable_llm.py | `test_lora_param_ratio_within_5_percent` + `test_lora_modules_not_empty` |
| T0-2 | StreamingEngine 完整流测试 | test_stream.py | `test_stream_done_message` + `test_stream_non_interrupt_full_flow` |
| T0-3 | VisionTokenizer PIL Image 测试 | test_tokenizer.py | `test_vision_tokenizer_pil_image_input` + `test_unified_encode_tuple_format` |

### P0 代码任务（GREEN 阶段）

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| P0-1 | 补充 LoRA 5% 测试 | test_pluggable_llm.py | T0-1 |
| P0-2 | 补充流式完整流测试 | test_stream.py | T0-2 |
| P0-3 | 补充 PIL Image 测试 | test_tokenizer.py | T0-3 |

---

## 执行命令

```bash
# 测试命令
/mnt/ssd/codespace/Tri-Transformer/backend/.venv/bin/pytest backend/tests/ --tb=short -q

# 增量 lint
cd /mnt/ssd/codespace/Tri-Transformer && git diff --name-only HEAD -- 'backend/*.py' | xargs -r backend/.venv/bin/flake8 --max-line-length=100
```

---

## 验收标准

- [ ] 所有原有 157 个测试不破坏
- [ ] LoRA 参数比 < 5% 精确测试通过
- [ ] StreamingEngine done 消息结构测试通过
- [ ] VisionTokenizer PIL Image 输入测试通过
- [ ] 新增测试后总数 ≥ 163（新增 ≥ 6 个用例）
