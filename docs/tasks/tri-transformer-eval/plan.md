# 任务清单：Tri-Transformer 目标函数与验证体系

## 概览

| 统计项 | 数量 |
|--------|------|
| P0 测试任务 | 5 |
| P0 代码任务 | 7 |
| P1 代码任务 | 2 |
| 总任务数 | 14 |
| 测试覆盖目标 | 90% |

---

## 执行顺序（TDD 流程）

### Phase 1：RED（先写测试）

所有测试任务必须先于对应代码任务执行。

| ID | 任务 | 文件 |
|----|------|------|
| T0-1 | RAGLoss 单元测试 | `eval/tests/test_rag_loss.py` |
| T0-2 | ControlAlignmentLoss 单元测试 | `eval/tests/test_control_alignment_loss.py` |
| T0-3 | HallucinationLoss 单元测试 | `eval/tests/test_hallucination_loss.py` |
| T0-4 | GroundTruthEngine 测试 | `eval/tests/test_ground_truth_engine.py` |
| T0-5 | EvalPipeline 集成测试 | `eval/tests/test_eval_pipeline.py` |

### Phase 2：GREEN（实现代码）

| ID | 任务 | 依赖 | 关键文件 |
|----|------|------|---------|
| P0-1 | 项目初始化 | - | `eval/requirements.txt`, `eval/pyproject.toml`, `eval/config.yaml` |
| P0-2 | RAGLoss 目标函数 | T0-1, P0-1 | `eval/loss/rag_loss.py` |
| P0-3 | ControlAlignmentLoss 目标函数 | T0-2, P0-2 | `eval/loss/control_alignment_loss.py` |
| P0-4 | HallucinationLoss + TotalLoss | T0-3, P0-3 | `eval/loss/hallucination_loss.py`, `eval/loss/total_loss.py` |
| P0-5 | GroundTruth 引擎 | T0-4, P0-1 | `eval/ground_truth/` 下所有文件 |
| P0-6 | Evaluation Pipeline | T0-5, P0-5 | `eval/pipeline/` 下所有文件 |
| P0-7 | 一键运行脚本 | P0-5, P0-6 | `eval/scripts/` 下所有文件 |

### Phase 3：P1 增强（可选）

| ID | 任务 | 依赖 |
|----|------|------|
| P1-1 | Docker 容器化部署 | P0-7 |
| P1-2 | GitHub Actions CI 集成 | P0-7 |

---

## 任务详情

### T0-1：RAGLoss 单元测试

**测试用例**：
- `test_recall_at_k_perfect`：完美检索 → Recall@K = 1.0
- `test_recall_at_k_zero`：零检索 → Recall@K = 0.0
- `test_ndcg_ordering`：完美排序 > 随机 > 逆序
- `test_rag_loss_differentiable`：`loss.backward()` 不报错，gradient norm > 0
- `test_weight_configuration`：α=0 → L_recall 不影响总损失

### T0-2：ControlAlignmentLoss 单元测试

**测试用例**：
- `test_contrastive_positive_vs_negative`：正样本相似度 > 负样本
- `test_nli_entailment_low_loss`：蕴含关系 → L_nli < 0.1
- `test_nli_contradiction_high_loss`：矛盾关系 → L_nli > 0.5
- `test_instruction_following_score`：遵循 → > 0.8，违反 → < 0.3

### T0-3：HallucinationLoss 单元测试

**测试用例**：
- `test_factual_loss_with_support`：有知识支撑 → L_fact < 0.2
- `test_factual_loss_without_support`：无支撑编造 → L_fact > 0.6
- `test_source_attribution_gap`：归因成功 vs 失败，损失差 > 0.3
- `test_abstention_calibration`：拒答 → 低损失；强行生成 → 高损失

### T0-4：GroundTruthEngine 测试

**测试用例**：
- `test_qa_generation_coverage`：每 1000 字 ≥ 5 QA 对
- `test_dual_model_consensus`：高相似度对识别准确
- `test_kg_triple_format`：Triple 结构完整
- `test_consistency_filter`：矛盾样本被过滤
- `test_fusion_difficulty_levels`：easy/medium/hard 均有分布
- `test_version_increment`：增量更新后版本号递增
- `test_jsonlines_export`：导出格式符合 Schema

### T0-5：EvalPipeline 集成测试

**测试用例**：
- `test_rag_metrics_range`：所有 RAG 指标在 [0,1]
- `test_generation_metrics_nonzero`：指标值非极端（不全 0 或 1）
- `test_hallucination_rate_calculation`：幻觉率计算正确（mock 验证）
- `test_ci_gate_pass`：满足阈值 → exit code 0
- `test_ci_gate_fail`：违反阈值 → exit code 1 + 原因
- `test_report_files_exist`：`eval_report.json` 和 `eval_report.md` 均生成
- `test_bootstrap_confidence_interval`：报告包含 `ci_lower` 和 `ci_upper`

---

## 目标函数设计摘要

### 总损失函数

```
L_total = w1 · L_rag + w2 · L_ctrl + w3 · L_hall

其中：
  L_rag  = α · Recall@K_loss + β · Coverage_loss + γ · NDCG_loss
  L_ctrl = λ₁ · L_contrastive + λ₂ · L_nli + λ₃ · L_inst
  L_hall = μ₁ · L_fact + μ₂ · L_attr + μ₃ · L_abstain
```

所有权重通过 `eval/config.yaml` 配置，默认值：
- `w1=0.3, w2=0.4, w3=0.3`
- `α=0.4, β=0.3, γ=0.3`
- `λ₁=0.4, λ₂=0.4, λ₃=0.2`
- `μ₁=0.4, μ₂=0.3, μ₃=0.3`

---

## CI Gate 阈值

| 指标 | 阈值 | 说明 |
|------|------|------|
| Hallucination Rate | < 0.05 | 幻觉率 < 5% |
| RAG Recall@5 | > 0.90 | 前5检索结果召回率 > 90% |
| BERTScore F1 | > 0.85 | 语义相似度 |

---

## 运行命令

```bash
# 构建 Ground Truth
python eval/scripts/build_ground_truth.py --docs_dir ./docs --output_dir ./eval/data

# 运行评估
python eval/scripts/run_eval.py --mode dataset --gt_file ./eval/data/gt_latest.jsonl

# CI 检查
python eval/scripts/ci_check.py
```
