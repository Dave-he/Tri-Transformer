# Requirement: Tri-Transformer Eval 工程完善开发

**任务 ID**: backend-server-eval  
**平台**: Backend/Python  
**创建时间**: 2026-03-30

## 背景

eval/ 目录已有完整骨架代码（loss 损失函数体系、ground_truth 构建引擎、pipeline 评估流水线、scripts CLI 脚本），但存在以下问题需修复并完善：

1. `RAGEvaluator` 错误 import `HallucinationLoss`（torch 污染）导致 4 项测试在无 torch 环境下 ERROR
2. `CIGate.check()` 中 `rag_recall_at_5 <= threshold` 方向 bug（应为 `<`）
3. `EvalPipeline.run()` 未调用 `ControlEvaluator` 和 `DialogCohesionEvaluator`
4. `total_loss.py` 缺少实现
5. 各 `__init__.py` 缺少统一导出
6. torch 相关测试缺少 skip 机制

## 功能需求

### FR-01: 修复 RAGEvaluator torch 循环依赖

RAGEvaluator 应与 torch 解耦，faithfulness 使用内置 token overlap 计算。

**验收标准**:
- RAGEvaluator 可在无 torch 环境下 import 和运行
- faithfulness/answer_relevancy/context_recall 指标计算正确

### FR-02: 修复 CIGate 阈值方向 Bug

`rag_recall_at_5 <= threshold` 逻辑反向，应改为 `< threshold`。

**验收标准**:
- recall_at_5 < threshold 时 CI FAIL
- recall_at_5 == threshold 时 CI PASS（边界正确）

### FR-03: 完善 EvalPipeline 集成所有 Evaluator

**验收标准**:
- run() 返回结果包含 instruction_following_rate, topic_consistency
- dialog_sessions=None 时不报错

### FR-04: 实现 total_loss.py 总损失聚合

`L_total = w1·L_rag + w2·L_ctrl + w3·L_hall`，默认权重 w1=0.3, w2=0.4, w3=0.3。

**验收标准**:
- TotalLoss 支持权重配置
- get_metrics() 返回各子损失值字典

### FR-05: 完善各模块 __init__.py 导出

**验收标准**:
- `from eval.loss import HallucinationLoss, RAGLoss, ControlAlignmentLoss, TotalLoss` 可用
- `from eval.pipeline import EvalPipeline, CIGate, ReportGenerator` 可用

### FR-06: torch 测试添加 skip 机制

**验收标准**:
- 无 torch 环境下 pytest 输出为 SKIP 非 ERROR
- 有 torch 环境下所有测试 PASS

### FR-07: 端到端集成测试

**验收标准**:
- EvalPipeline.run() 完整运行并返回预期字段
- CIGate 阻断和放行场景均有覆盖

### FR-08: CLI 脚本可运行验证

**验收标准**:
- build_ground_truth.py、run_eval.py、ci_check.py 功能可测试

## 非功能需求

- 无 torch 环境下 pytest pass 率 ≥ 90%（torch 测试允许 skip）
- 模块 import 无副作用
- eval/ 与 backend/ 解耦

## 约束

- 不修改现有算法逻辑，只做修复和补全
- 保持现有代码风格（无注释）
