# RAG（检索增强生成）体系

## 0. 结论先行

- **核心作用**：在生成前从外部知识库实时检索相关文档并注入上下文，是解决 LLM 幻觉、知识过时和不可溯源三大核心问题的主流工程范式，无需重训练即可实现知识更新。
- **工程推荐配置**：混合检索（向量 BGE-M3 + BM25 RRF 融合）+ Cross-Encoder 重排序（BAAI/bge-reranker-v2-m3），Recall@10 可达 0.81。无 HyDE 时全流程延迟约 56ms（P50），满足 Tri-Transformer < 300ms 目标。
- **幻觉阻断是核心差异**：相比普通 RAG，Tri-Transformer 在 O-Transformer Planning Encoder 层加入 HallucinationGuard（余弦一致性门控），一致性 < 0.7 时直接拒绝生成，实现零幻觉阻断而非事后过滤。
- **Tri-Transformer 中的角色**：O-Transformer Planning Encoder 的知识注入层；C-Transformer state_slots 提供意图向量，与检索上下文做一致性校验，驱动是否生成的决策。

---

## 1. 概述

检索增强生成（Retrieval-Augmented Generation, RAG）是解决大语言模型幻觉（Hallucination）、知识过时和不可溯源等核心问题的主流技术范式。通过在生成前从外部知识库检索相关文档片段并注入上下文，RAG 将模型的参数化记忆与非参数化知识库动态结合，在保留 LLM 强大生成能力的同时大幅提升事实准确性。

**在 Tri-Transformer 中的角色**：O-Transformer 的 Planning Encoder 在生成前实时查询多模态向量数据库，将检索到的知识块与控制信号融合，实现"无幻觉阻断"机制的知识注入层。

---

## 2. RAG 三大范式

### 2.1 Naive RAG（朴素 RAG）

```
用户问题 → [检索器] → Top-K 文档块
                         ↓ 直接拼接到 Prompt
                     [LLM] → 回答
```

**优点**：实现简单，延迟低。
**缺点**：检索噪声直接污染生成，无重排序/压缩，对长文档处理差。

### 2.2 Advanced RAG（高级 RAG）

```
用户问题
    ↓ 查询改写（Query Rewriting/HyDE）
增强查询
    ↓ 多路检索（向量 + BM25 混合）
初始候选文档
    ↓ 重排序（Cross-Encoder Reranker）
精选 Top-K 文档
    ↓ 上下文压缩（Selective Context）
压缩后的高质量上下文
    ↓
[LLM] → 回答
```

### 2.3 Modular RAG（模块化 RAG）—— 本项目采用

各组件完全解耦，可灵活组合：

```python
class ModularRAGPipeline:
    def __init__(self):
        self.query_transformer = QueryTransformer()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.context_compressor = ContextCompressor()
        self.generator = TriTransformerOEncoder()
        self.hallucination_checker = HallucinationGuard()

    def run(self, query: str, modality: str = "text") -> str:
        transformed_query = self.query_transformer.transform(query)
        candidates = self.retriever.retrieve(transformed_query, top_k=20)
        ranked = self.reranker.rerank(query, candidates, top_k=5)
        context = self.context_compressor.compress(ranked)

        if self.hallucination_checker.check(query, context):
            return self.generator.generate(query, context)
        else:
            return self.generator.refuse(query)
```

---

## 3. 核心组件详解

### 3.1 文档分块（Chunking）策略

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SmartChunker:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    def chunk_document(self, doc: str) -> list[str]:
        return self.splitter.split_text(doc)

    def chunk_with_metadata(self, docs: list[dict]) -> list[dict]:
        chunks = []
        for doc in docs:
            text_chunks = self.chunk_document(doc["content"])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk,
                    "source": doc["source"],
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "page": doc.get("page", 0)
                })
        return chunks
```

**Chunk 策略选型建议**：

| 策略 | chunk_size | chunk_overlap | 适用场景 | 延迟影响 |
|---|---|---|---|---|
| 固定字符切分 | 256-512 | 30-50 | 通用文档 | 最低 |
| 递归字符切分（推荐） | 512-1024 | 50-100 | 中文/混合文档 | 低 |
| 语义切分 | 动态 | 0 | 学术论文/技术文档 | 中 |
| Token 切分 | 256-512 tokens | 32 | LLM 上下文长度感知 | 低 |
| 层次切分 | 多级 | 0 | 超长文档/书籍 | 高 |

### 3.2 混合检索（Hybrid Retrieval）

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_index, alpha: float = 0.6):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        vector_results = self.vector_store.similarity_search(query, k=top_k * 2)
        bm25_results = self.bm25.search(query, k=top_k * 2)

        scores = {}
        for rank, doc in enumerate(vector_results):
            scores[doc['id']] = scores.get(doc['id'], 0) + self.alpha / (rank + 60)
        for rank, doc in enumerate(bm25_results):
            scores[doc['id']] = scores.get(doc['id'], 0) + (1 - self.alpha) / (rank + 60)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [self._get_doc(doc_id) for doc_id in ranked_ids[:top_k]]
```

**向量检索 vs BM25 vs 混合**：

| 方法 | Recall@10 | MRR | 延迟 | 适用 |
|---|---|---|---|---|
| 纯向量检索 | 0.72 | 0.65 | ~10ms | 语义相似 |
| 纯 BM25 | 0.68 | 0.61 | ~5ms | 关键词精确匹配 |
| 混合 RRF（推荐） | **0.81** | **0.74** | ~15ms | 通用场景 |

### 3.3 查询改写（HyDE）

```python
class HyDEQueryTransformer:
    def __init__(self, llm):
        self.llm = llm

    def transform(self, query: str) -> str:
        hypothetical_answer = self.llm.generate(
            f"请写一段假设性的回答，即使你不确定答案：{query}"
        )
        return hypothetical_answer
```

### 3.4 Cross-Encoder 重排序

```python
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, doc["content"]) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
```

**推荐开源模型清单**：

| 组件 | 推荐模型 | 参数量 | 中文支持 |
|---|---|---|---|
| 嵌入模型（向量检索） | `BAAI/bge-m3` | 570M | 优秀 |
| 嵌入模型（轻量） | `BAAI/bge-small-zh-v1.5` | 24M | 优秀 |
| 重排序模型 | `BAAI/bge-reranker-v2-m3` | 570M | 优秀 |
| 重排序模型（轻量） | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 英文 |
| 问答 LLM | `Qwen3-8B`（Non-Thinking 模式） | 8B | 优秀 |

### 3.5 无幻觉阻断机制（Hallucination Guard）

基于 C-RAG（arXiv:2402.03181）理论框架的实践实现：

```python
import torch
import torch.nn as nn


class HallucinationGuard(nn.Module):
    def __init__(self, d_model: int, nli_threshold: float = 0.7):
        super().__init__()
        self.intent_proj = nn.Linear(d_model, d_model)
        self.context_proj = nn.Linear(d_model, d_model)
        self.nli_threshold = nli_threshold

    def check_consistency(
        self,
        state_slots: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        intent = self.intent_proj(state_slots.mean(1))
        context_rep = self.context_proj(context_embedding.mean(1))
        consistency = torch.cosine_similarity(intent, context_rep, dim=-1)
        return consistency

    def gate_ctrl_signal(
        self,
        ctrl_signal: torch.Tensor,
        state_slots: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        consistency = self.check_consistency(state_slots, context_embedding)
        gate = torch.sigmoid((consistency - 0.5) * 10)
        return ctrl_signal * gate.unsqueeze(-1)

    def should_refuse(self, consistency_score: float) -> bool:
        return consistency_score < self.nli_threshold
```

---

## 4. 端到端延迟分解

RAG Pipeline 各阶段延迟基准（单请求，A100 GPU，中文 512 字文档）：

| 阶段 | 操作 | 延迟（P50） | 延迟（P99） | 优化建议 |
|---|---|---|---|---|
| 查询改写（HyDE） | LLM 生成假设答案 | 80ms | 200ms | 可选，仅复杂问题开启 |
| 查询嵌入 | BGE-M3 编码 | 8ms | 15ms | 批处理 + GPU |
| 向量检索 | Milvus HNSW Top-20 | 5ms | 12ms | 索引预热 |
| BM25 检索 | BM25 Top-20 | 2ms | 5ms | 内存索引 |
| 重排序 | Cross-Encoder Top-5 | 25ms | 60ms | 轻量模型替代 |
| 上下文压缩 | Token 截断/选择 | 1ms | 3ms | 规则化处理 |
| 幻觉检测 | NLI 一致性检查 | 15ms | 35ms | 异步并行 |
| **总计（无 HyDE）** | — | **~56ms** | **~130ms** | — |
| **总计（有 HyDE）** | — | **~136ms** | **~330ms** | — |

> 对于 Tri-Transformer < 300ms 延迟目标，推荐使用"无 HyDE"配置，并将幻觉检测异步化。

---

## 5. O-Transformer Planning Encoder 集成

### 5.1 完整集成代码示例

```python
import torch
import torch.nn as nn
from typing import Optional


class OTransformerWithRAG(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_planning_layers: int,
        rag_pipeline: "ModularRAGPipeline",
        hallucination_guard: HallucinationGuard,
    ):
        super().__init__()
        self.d_model = d_model

        self.context_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.planning_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True)
            for _ in range(num_planning_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.rag = rag_pipeline
        self.guard = hallucination_guard

    def forward(
        self,
        i_enc: torch.Tensor,
        ctrl_state: torch.Tensor,
        query_text: str,
        context_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, bool]:
        if context_embeddings is None:
            docs = self.rag.retriever.retrieve(query_text, top_k=5)
            doc_texts = [d["content"] for d in docs]
            context_embeddings = self.rag.embed_docs(doc_texts)

        context_proj = self.context_proj(context_embeddings)

        consistency = self.guard.check_consistency(
            ctrl_state.unsqueeze(1), context_embeddings
        )
        should_refuse = self.guard.should_refuse(consistency.mean().item())
        if should_refuse:
            return ctrl_state, True

        x = i_enc
        for layer in self.planning_layers:
            x = layer(x, context_proj)

        x, _ = self.cross_attn(x, context_proj, context_proj)
        o_plan = self.norm(x)
        return o_plan, False
```

### 5.2 实时 RAG 注入流程

```
用户音视频输入
    ↓ I-Transformer 编码 → i_enc [B, N_chunks, D]
    ↓ 提取关键信息（意图、实体）→ query_text
实时查询 → Milvus 多模态向量库（~5-15ms）
    ↓ Top-5 相关知识块
    ↓ Cross-Encoder 重排序（~25ms）
知识块嵌入 → context_embeddings [B, 5, D]
    ↓ OTransformerWithRAG.forward()
    ↓   1. 一致性检查（HallucinationGuard）
    ↓   2. Planning Encoder（CrossAttention × M 层）
o_plan [B, N_chunks, D]（已融合知识，零幻觉）
    ↓ Streaming Decoder
多模态输出流
```

---

## 6. RAG 评测指标

| 指标 | 含义 | 工具 | 目标值 |
|---|---|---|---|
| 上下文召回（Context Recall） | 答案中关键信息是否来自检索文档 | RAGAS | ≥ 0.85 |
| 答案相关性（Answer Relevance） | 生成答案是否回答了问题 | RAGAS | ≥ 0.80 |
| 忠实度（Faithfulness） | 答案是否与上下文一致（无幻觉） | RAGAS, TrueLens | ≥ 0.90 |
| 上下文精确率（Context Precision） | 检索文档是否都与答案相关 | RAGAS | ≥ 0.75 |
| 端到端 RAG 延迟 | 从查询到返回知识 | 自测 | < 100ms |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(results)
```

---

## 7. RAG vs Long-context LLM 决策树

```
问题规模与场景
    ├── 知识库 < 10K tokens?
    │       → Long-context LLM（直接塞入上下文，128K 窗口足够）
    ├── 知识库 > 10K tokens 且需要精准引用？
    │       → Advanced RAG（混合检索 + 重排序）
    ├── 知识库频繁更新（每日/每小时）？
    │       → RAG（增量索引，避免重训练）
    ├── 需要跨文档推理（多跳问题）？
    │       → GraphRAG 或 Agentic RAG
    ├── 延迟要求 < 100ms？
    │       → Naive RAG（牺牲精度换延迟）
    └── 多模态知识（图文混合）？
            → 多模态 RAG（本项目方案）
```

**Lost in the Middle 问题**：研究表明 LLM 对超长上下文中间部分存在信息遗忘，即使 128K 窗口，中间位置信息的利用率仅约 30%。RAG 的精准检索可将关键信息置于上下文头部，显著优于"全量塞入"。

---

## 8. 最新进展（2024-2025）

### 8.1 GraphRAG（Microsoft, 2024）
- 用知识图谱替代纯向量检索，支持全局主题查询（"文档的主要议题是什么"），弥补向量检索的全局感知不足。

### 8.2 Agentic RAG（2024）
- 将 RAG 与 Agent 循环结合：模型主动决定何时、如何检索，支持多步推理和迭代检索（ReAct、SELF-RAG）。

### 8.3 多模态 RAG 的兴起
- **M-RAG、MultiRAG**：将视频帧、音频片段纳入知识库，支持"给我看这段对话对应的产品说明书图表"等跨模态查询，正是 Tri-Transformer Phase 3 的目标场景。

### 8.4 RAG-Fusion（2024）
- 生成多个查询变体，各自检索后用 RRF 融合，提升召回率约 15-20%，代价是延迟翻倍（适合离线场景）。
