# 需求文档：Tri-Transformer 目标函数与验证体系设计

## 背景与核心目标

基于《Tri-Transformer 可控对话与 RAG 知识库增强系统》PRD，本需求聚焦于设计：

1. **目标函数体系**：覆盖 RAG 检索、C 分支控制对齐、幻觉抑制三类核心损失函数
2. **Agent 自主 Ground Truth 发现引擎**：无需人工标注，自动从文档库构建评估数据集
3. **自动化 Evaluation Pipeline**：端到端可量化验证框架，集成 CI/CD 质量门禁

---

## 功能模块

### 模块一：目标函数体系（FR-01 ~ FR-03）

#### FR-01：RAG 检索质量目标函数
- **检索相关性损失**：Recall@K（K=1,3,5,10）
- **信息覆盖损失**：F1-Token Overlap / BERTScore
- **排序一致性损失**：NDCG + MRR
- 目标函数可微分，支持端到端梯度传播

#### FR-02：Tri-Transformer 控制对齐损失（C 分支核心）
- **控制信号对齐损失**：基于对比学习（Contrastive Learning），正负样本对训练
- **知识一致性损失**：NLI 模型检测生成内容与检索知识的蕴含/矛盾关系
- **指令遵循损失**：分类（是否遵循）+ 回归（遵循程度）双形式
- 三类损失通过可配置权重 `α·L_ctrl + β·L_know + γ·L_inst` 加权合并

#### FR-03：幻觉抑制损失函数
- **事实幻觉损失**：FactScore / 基于检索的事实核查
- **来源归因损失**：Attention Weight 分析，片段级事实追溯
- **拒答校准损失**：置信度阈值可配置，知识缺失时惩罚强行生成
- 聚合指标：**幻觉率（Hallucination Rate）**

---

### 模块二：Agent 自主 Ground Truth 构建（FR-04 ~ FR-05）

#### FR-04：Ground Truth 自主发现引擎

Agent 通过以下四条路径自主构建 Ground Truth，**无需人工标注**：

| 路径 | 方法 | 质量等级 |
|------|------|---------|
| 文档内置事实挖掘 | 自动生成问答对（每 1000 字 ≥ 5 对） | Silver |
| 双大模型交叉验证 | 两模型共识采样，准确率 ≥ 85% | Gold |
| 知识图谱 Triple 提取 | (主体, 关系, 客体) 结构化 Ground Truth | Silver |
| 对比一致性验证 | 多轮追问检验自洽性，过滤矛盾样本 | Gold |

#### FR-05：多源融合与质量评估
- **来源权重**：人工标注 > 双模型共识 > 单模型生成
- **难度分级**：Easy / Medium / Hard，支持分级评估
- **导出格式**：JSON Lines / HuggingFace Dataset
- **版本控制**：支持历史版本回溯

---

### 模块三：自动化 Evaluation Pipeline（FR-06 ~ FR-07）

#### FR-06：多维度评估指标体系

| 评估维度 | 指标 | 工具 |
|----------|------|------|
| RAG 质量 | Faithfulness, Answer Relevancy, Context Recall | RAGAS |
| 生成质量 | BLEU-4, ROUGE-L, BERTScore F1, METEOR | HF evaluate |
| 控制性 | 指令遵循率, 主题一致性, 风格一致性 | 自研 |
| 幻觉率 | FactScore, 知识库覆盖率, 来源追溯成功率 | 自研 + FactScore |
| 对话连贯性 | 多轮一致性, 上下文保持率 | 自研 |

- Pipeline 支持单条 / 批量 / 数据集级评估
- 每个指标提供 Bootstrap 置信区间
- 自动生成 JSON + Markdown 双格式报告

#### FR-07：CI Gate 与实时监控
- 每次训练后自动触发 benchmark
- **CI 通过门槛**：
  - Hallucination Rate < 5%
  - RAG Recall@5 > 90%
  - BERTScore F1 > 0.85
- 可视化 Dashboard：指标趋势 + 多维钻取
- 告警支持邮件 / Webhook

---

## 验收标准

| # | 标准 | 量化目标 |
|---|------|---------|
| 1 | Ground Truth 自动构建 | ≥ 500 条高质量样本，无人工介入 |
| 2 | 目标函数单元测试 | 覆盖率 100%，数值正确性验证 |
| 3 | Evaluation Pipeline | 一键运行，< 5min / 1000 样本 |
| 4 | CI Gate | 自动阻断性能退化版本 |
| 5 | Docker 部署 | 一键启动，无手动配置 |

---

## 技术约束

- **语言**：Python 3.10+
- **框架**：PyTorch, HuggingFace Transformers, LlamaIndex
- **向量库**：Chroma (dev) / Milvus (prod)
- **评估工具**：RAGAS, deepeval, HF evaluate
- **约束**：完全离线可运行，无商业 API 依赖
