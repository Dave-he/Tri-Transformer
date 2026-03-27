# 研究发现

## 背景
Tri-Transformer 是融合三分支架构（I-编码/C-控制/O-解码）与 RAG 知识库的可控对话系统。
本次任务聚焦于：目标函数体系 + Agent 自主 Ground Truth 发现 + Evaluation Pipeline。

## 关键发现

### 发现 1: 系统三大核心评估维度
- **来源**: PRD 文档 §3.1 功能模块
- **内容**: 
  1. RAG 检索质量（检索相关性、信息覆盖、排序一致性）
  2. 生成可控性（指令遵循、知识一致性、风格控制）
  3. 幻觉抑制（事实核查、来源归因、拒答校准）
- **影响**: 三类目标函数各自独立设计，通过加权合并成总损失
- **日期**: 2026-03-27

### 发现 2: Agent 自主 GT 构建的四条路径
- **来源**: 需求分析 FR-04
- **内容**:
  1. 文档内置事实挖掘（Document QA Generation）
  2. 双大模型交叉验证（Dual-LLM Cross Validation）
  3. 知识图谱 Triple 提取（KG Triple Extraction）
  4. 对比一致性验证（Contrastive Consistency Check）
- **影响**: 需要设计统一的 Ground Truth Schema 兼容四种来源
- **日期**: 2026-03-27

### 发现 3: 技术栈约束
- **来源**: 需求约束 §constraints
- **内容**: Python 3.10+, PyTorch, RAGAS, deepeval, HF evaluate, BGE 系列模型
- **影响**: 所有组件需本地可运行，无商业 API 依赖
- **日期**: 2026-03-27

### 发现 4: CI Gate 指标
- **来源**: 需求 FR-07
- **内容**: Hallucination Rate < 5%, RAG Recall@5 > 90%, BERTScore F1 > 0.85
- **影响**: 这三个指标是 CI 流水线的硬性质量门禁
- **日期**: 2026-03-27

## 技术笔记

### 目标函数设计要点
- C 分支控制对齐损失需要 Contrastive Learning，正样本=遵循控制信号的生成，负样本=违反控制信号的生成
- 知识一致性损失基于 NLI：生成内容与检索知识的关系应为"蕴含(entailment)"，而非"矛盾(contradiction)"
- 拒答校准：当 RAG Top-K 的最高相关度分数 < 阈值时，模型应输出"无法回答"而非强行生成

### Ground Truth 构建要点
- 文档 QA 生成：使用 LLM（Qwen/Llama）对文档段落生成问题，再用另一个 LLM 回答
- 双模型验证：两个模型回答相同，则为 Gold；只有一个模型回答，则为 Silver
- Triple 提取：使用 spaCy + 开源 IE 模型提取实体关系三元组

## 参考资料
- RAGAS: https://github.com/explodinggradients/ragas
- FactScore: https://github.com/shmsw25/FActScoring
- deepeval: https://github.com/confident-ai/deepeval
- BGE Reranker: https://huggingface.co/BAAI/bge-reranker-large
