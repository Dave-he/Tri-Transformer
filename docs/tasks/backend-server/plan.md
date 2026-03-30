# 开发任务清单: Tri-Transformer Eval 工程 Bug 修复与完善

**来源**: docs/tasks/backend-server/tech-solution.yaml  
**测试覆盖目标**: 100%（P0 必须覆盖）

## 任务概览

| ID | 类型 | 优先级 | 描述 | 依赖 |
|----|------|--------|------|------|
| T0-1 | test | P0 | torch importorskip - test_hallucination_loss | - |
| T0-2 | test | P0 | torch importorskip - test_rag_loss | - |
| T0-3 | test | P0 | torch importorskip - test_control_alignment_loss | - |
| T1-1 | test | P0 | RAGEvaluator 无 torch 可运行 | - |
| T1-2 | test | P0 | CIGate 阈值边界 | - |
| T2-1 | test | P0 | EvalPipeline 集成所有 Evaluator | - |
| T3-1 | test | P1 | loss/__init__.py 导出 | - |
| P0-1 | code | P0 | 修复 RAGEvaluator torch 依赖 | T1-1 |
| P0-2 | code | P0 | 修复 CIGate 阈值逻辑 | T1-2 |
| P0-3 | code | P0 | 修复 EvalPipeline 集成 | T2-1 |
| P0-4 | code | P0 | torch 测试 importorskip | T0-1, T0-2, T0-3 |
| P1-1 | code | P1 | eval/loss/__init__.py 导出 | T3-1 |
| P1-2 | code | P1 | eval/pipeline/__init__.py 导出 | - |

## TDD 执行顺序

### RED 阶段（先写/更新测试）
1. T0-1: 在 test_hallucination_loss.py 每个 class 首行加 importorskip → 验证 SKIP
2. T0-2: 在 test_rag_loss.py 每个 class 首行加 importorskip → 验证 SKIP
3. T0-3: 在 test_control_alignment_loss.py 每个 class 首行加 importorskip → 验证 SKIP
4. T1-1: 在 test_eval_pipeline.py 添加 TestRAGEvaluatorNoBiasTest → 验证 ERROR（改前）
5. T1-2: 在 test_eval_pipeline.py 添加 TestCIGateBoundary → 验证 FAIL（改前边界测试失败）
6. T2-1: 在 test_eval_pipeline.py 添加 TestEvalPipelineIntegration → 验证 FAIL（缺字段）

### GREEN 阶段（修复代码）
7. P0-4: 修改三个 loss 测试文件，加 importorskip
8. P0-1: 修改 rag_evaluator.py，移除 torch import，内联实现
9. P0-2: 修改 ci_gate.py，`<=` 改为 `<`
10. P0-3: 修改 eval_pipeline.py，集成 ControlEvaluator + DialogEvaluator
11. P1-1: 填充 eval/loss/__init__.py
12. P1-2: 填充 eval/pipeline/__init__.py

## 验收标准

- `pytest eval/tests/` 0 ERROR
- CIGate: recall=0.90, threshold=0.90 → PASS
- CIGate: recall=0.89, threshold=0.90 → FAIL
- EvalPipeline.run() 返回包含 instruction_following_rate

## 测试命令
```bash
.venv/bin/pytest eval/tests/ -v --tb=short
```
