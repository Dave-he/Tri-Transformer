# RAG（检索增强生成）体系

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

### 3.2 混合检索（Hybrid Retrieval）

```python
class HybridRetriever:
    """向量检索（语义）+ BM25（关键词）混合，RRF 融合排序"""
    def __init__(self, vector_store, bm25_index, alpha: float = 0.5):
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

### 3.3 查询改写（HyDE：假设文档嵌入）

```python
class HyDEQueryTransformer:
    """生成假设性回答，用假设回答的嵌入而非原始问题嵌入检索"""
    def __init__(self, llm):
        self.llm = llm

    def transform(self, query: str) -> str:
        hypothetical_answer = self.llm.generate(
            f"请写一段假设性的回答，即使你不确定答案：{query}"
        )
        return hypothetical_answer
```

### 3.4 无幻觉阻断机制（Hallucination Guard）

基于 C-RAG（arXiv:2402.03181）理论框架的实践实现：

```python
class HallucinationGuard:
    def __init__(self, nli_model, threshold: float = 0.7):
        self.nli = nli_model
        self.threshold = threshold

    def check(self, query: str, context: list[str], generated: str = None) -> bool:
        """检查生成内容是否与检索上下文矛盾"""
        if generated is None:
            return True

        contradiction_score = self.nli.predict_contradiction(context, generated)
        if contradiction_score > self.threshold:
            return False
        return True

    def gate_generation(self, ctrl_signal: torch.Tensor,
                         context_embedding: torch.Tensor,
                         state_slots: torch.Tensor) -> torch.Tensor:
        """
        C-Transformer 的内部幻觉检测：
        比较状态槽内的内容意图与 RAG 上下文的一致性
        """
        intent = self.intent_proj(state_slots.mean(1))
        context_rep = self.context_proj(context_embedding.mean(1))
        consistency = torch.cosine_similarity(intent, context_rep, dim=-1)
        gate = torch.sigmoid((consistency - 0.5) * 10)
        return ctrl_signal * gate.unsqueeze(-1)
```

---

## 4. 多模态 RAG

### 4.1 多模态文档索引

```python
class MultimodalDocumentIndex:
    """支持文本、图像、音频的多模态知识库"""
    def __init__(self, text_embedder, image_embedder, audio_embedder, vector_db):
        self.embedders = {
            "text": text_embedder,
            "image": image_embedder,
            "audio": audio_embedder
        }
        self.vector_db = vector_db

    def index_document(self, doc: dict):
        if doc["type"] == "text":
            embedding = self.embedders["text"].embed(doc["content"])
        elif doc["type"] == "image":
            embedding = self.embedders["image"].embed(doc["image_path"])
        elif doc["type"] == "audio":
            embedding = self.embedders["audio"].embed(doc["audio_path"])

        self.vector_db.insert({
            "id": doc["id"],
            "embedding": embedding,
            "type": doc["type"],
            "content": doc["content"],
            "metadata": doc.get("metadata", {})
        })
```

### 4.2 实时 RAG 注入流程（Tri-Transformer 集成）

```
用户音视频输入
    ↓ I-Transformer 编码 → i_enc
    ↓ 提取关键信息（意图、实体）
实时查询 → Milvus 多模态向量库
    ↓ Top-5 相关知识块
知识块嵌入 rag_context
    ↓ O-Transformer Planning Encoder
    CrossAttention(x, rag_context)
    ↓ 验证一致性（HallucinationGuard）
o_plan（规划表征，已融合知识）
    ↓ Streaming Decoder
多模态输出流（零幻觉）
```

---

## 5. RAG 评测指标

| 指标 | 含义 | 工具 |
|---|---|---|
| 上下文召回（Context Recall） | 答案中关键信息是否来自检索文档 | RAGAS |
| 答案相关性（Answer Relevance） | 生成答案是否回答了问题 | RAGAS |
| 忠实度（Faithfulness） | 答案是否与上下文一致（无幻觉） | RAGAS, TrueLens |
| 上下文精确率（Context Precision） | 检索文档是否都与答案相关 | RAGAS |
| 端到端准确率 | 答案是否正确 | 人工/F1/ROUGE |

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

## 6. 最新进展（2024-2025）

### 6.1 GraphRAG（Microsoft, 2024）
- 用知识图谱替代纯向量检索，支持全局主题查询（"文档的主要议题是什么"），弥补向量检索的全局感知不足。

### 6.2 Long-context RAG 的挑战
- 模型上下文窗口扩展（128K+）后，RAG 的必要性受到质疑。但研究表明 LLM 对超长上下文中间部分存在"迷失现象（Lost in the Middle）"，RAG 的精准检索仍有价值。

### 6.3 Agentic RAG（2024）
- 将 RAG 与 Agent 循环结合：模型主动决定何时、如何检索，支持多步推理和迭代检索（ReAct、SELF-RAG）。

### 6.4 多模态 RAG 的兴起
- **M-RAG、MultiRAG**：将视频帧、音频片段纳入知识库，支持"给我看这段对话对应的产品说明书图表"等跨模态查询，正是 Tri-Transformer Phase 3 的目标场景。
