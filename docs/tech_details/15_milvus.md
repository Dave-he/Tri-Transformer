# Milvus（多模态向量数据库）

## 0. 结论先行

- **核心定位**：云原生向量数据库，十亿级向量毫秒级 ANN 检索，原生支持混合搜索（向量 + 标量过滤），是多模态 RAG 系统知识库底座的事实标准。
- **工程推荐**：开发阶段用 Milvus Lite（单文件，无需 Docker）；生产环境用 Standalone（单节点）或 Distributed（Kubernetes 多节点）；索引策略：< 100 万向量用 HNSW（延迟最低），> 1 亿向量用 DiskANN（存储高效），混合检索用稀疏向量 + BM25。
- **关键参数**：embedding 维度与模型对齐（text: 1024/1536，vision: 1024，audio: 512）；相似度度量优先用 `COSINE`（归一化向量）；批量插入建议 batch_size=1000-5000。
- **Tri-Transformer 中的角色**：存储多模态知识库（文本/图像/音频片段）的嵌入向量，支持 O-Transformer Planning Encoder 生成前的跨模态 RAG 查询；状态槽（State Slots）的持久化存储后端候选。

---

## 1. 概述

Milvus 是由 Zilliz 开源、Linux Foundation AI & Data 托管的云原生向量数据库，专为海量向量相似度搜索而设计。它支持十亿级向量的毫秒级近似最近邻（ANN）检索，是目前最活跃的开源向量数据库之一（GitHub 43.5K stars）。

Milvus 支持多种数据类型（浮点向量、二进制向量、稀疏向量、全文检索），并原生支持混合搜索（向量相似度 + 标量字段过滤），是多模态 RAG 系统的理想知识库底座。

**在 Tri-Transformer 中的角色**：存储多模态知识库（文档文本、图表图像、音频片段）的嵌入向量，O-Transformer Planning Encoder 生成前实时查询，返回最相关的知识块用于注入。

---

## 2. 系统架构

### 2.1 Milvus 2.x 分布式架构

```
┌─────────────────────────────────────────────────────┐
│                    SDK / REST API                    │
└───────────────────────┬─────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                   Proxy Layer                        │
│   请求路由 + 负载均衡 + 鉴权                          │
└──────┬──────────────────────────────────────────────┘
       ↓                    ↓                    ↓
┌─────────────┐    ┌───────────────┐    ┌──────────────┐
│  Query Node │    │  Data Node    │    │  Index Node  │
│ （执行查询） │    │ （数据写入）   │    │ （构建索引）  │
└─────────────┘    └───────────────┘    └──────────────┘
       ↓                    ↓
┌─────────────────────────────────────────────────────┐
│              Object Storage（MinIO/S3）               │
│         + etcd（元数据）+ Pulsar/Kafka（消息队列）     │
└─────────────────────────────────────────────────────┘
```

**核心设计原则**：
- **存算分离**：存储（Object Storage）与计算（Query/Data Node）完全解耦，支持独立弹性扩缩容。
- **流批一体**：实时写入与批量索引并行，保证低延迟写入的同时维持高质量索引。

---

## 3. 索引类型详解

### 3.1 主要索引类型对比

| 索引 | 算法 | 内存占用 | 查询速度 | 准确率 | 适用场景 |
|---|---|---|---|---|---|
| **FLAT** | 暴力搜索 | 高 | 慢 | 100% | 小数据集（<1M）、精确召回 |
| **IVF_FLAT** | 倒排文件+暴力 | 中 | 中 | 高 | 中等数据集 |
| **IVF_PQ** | 倒排文件+乘积量化 | 低 | 快 | 中 | 大数据集、内存受限 |
| **HNSW** | 层次化可导航小世界 | 高 | 极快 | 高 | 高召回率+高速度（推荐） |
| **DiskANN** | 基于SSD的图索引 | 极低 | 快 | 高 | 超大规模（10B+向量） |
| **SCANN** | 各向异性量化 | 中 | 极快 | 高 | Google 生产级场景 |

### 3.2 HNSW 原理

HNSW（Hierarchical Navigable Small World）构建层次图结构：

```
第2层（稀疏）: [0]────────────[5]
                \              /
第1层（中等）:  [0]──[1]──[3]──[5]──[7]
                              |
第0层（密集）:  [0][1][2][3][4][5][6][7][8][9]

查询: 从顶层开始，贪婪选择最近邻，逐层下降到底层精确搜索
```

**特点**：查询时间复杂度 $O(\log N)$，构建时间较长但查询极快，是 Tri-Transformer 实时 RAG 的首选索引。

---

## 4. 使用方法

### 4.1 安装与启动

```bash
pip install pymilvus

docker run -d --name milvus-standalone \
    -p 19530:19530 \
    -p 9091:9091 \
    -v /mnt/milvus:/var/lib/milvus \
    milvusdb/milvus:v2.4.0 \
    milvus run standalone
```

### 4.2 创建多模态知识库

```python
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

connections.connect(host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

schema = CollectionSchema(fields, description="Tri-Transformer 多模态知识库")
collection = Collection("tri_transformer_kb", schema)

index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)
collection.load()
print("知识库创建完成")
```

### 4.3 知识库写入

```python
def insert_text_document(collection: Collection, text: str, source: str, embedder):
    embedding = embedder.encode(text)
    collection.insert([{
        "modality": "text",
        "source": source,
        "content": text[:4096],
        "embedding": embedding.tolist()
    }])

def insert_image(collection: Collection, image_path: str, source: str,
                 caption: str, vision_embedder):
    embedding = vision_embedder.encode_image(image_path)
    collection.insert([{
        "modality": "image",
        "source": source,
        "content": caption,
        "embedding": embedding.tolist()
    }])
```

### 4.4 实时 RAG 查询

```python
def query_knowledge(collection: Collection, query_embedding: list,
                    top_k: int = 5, modality_filter: str = None) -> list:
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    expr = None
    if modality_filter:
        expr = f'modality == "{modality_filter}"'

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["modality", "source", "content"]
    )

    knowledge_chunks = []
    for hit in results[0]:
        knowledge_chunks.append({
            "content": hit.entity.get("content"),
            "source": hit.entity.get("source"),
            "modality": hit.entity.get("modality"),
            "score": hit.score
        })
    return knowledge_chunks
```

### 4.5 与 O-Transformer 集成

```python
class TriTransformerRAGConnector:
    def __init__(self, collection, text_embedder, vision_embedder):
        self.collection = collection
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.rag_proj = nn.Linear(1024, 1024)

    def retrieve_and_embed(self, query_state: torch.Tensor,
                           modality: str = None) -> torch.Tensor:
        query_np = query_state.mean(1).detach().cpu().numpy()[0].tolist()
        chunks = query_knowledge(self.collection, query_np, top_k=5,
                                 modality_filter=modality)

        if not chunks:
            return torch.zeros(1, 1, 1024, device=query_state.device)

        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.text_embedder.encode(chunk_texts)
        rag_tensor = torch.tensor(embeddings, device=query_state.device).unsqueeze(0)
        return self.rag_proj(rag_tensor)
```

---

## 5. 最新进展（2024-2025）

### 5.1 Milvus 2.4（2024）
- **稀疏向量支持**：原生支持 BM25 稀疏向量，支持混合稠密+稀疏搜索，无需额外 BM25 引擎。
- **GPU 加速索引**：GPU_IVF_FLAT、GPU_CAGRA 索引，构建与查询速度大幅提升。
- **全文搜索（BM25）**：内置关键词检索能力，真正的"一库解决向量+关键词"混合搜索。

### 5.2 Milvus Lite（2024）
- 单文件本地部署版本，适用于开发测试，API 与完整版兼容，可一键迁移至分布式部署。

### 5.3 竞品对比

| 产品 | 类型 | 优势 | 劣势 |
|---|---|---|---|
| **Milvus** | 开源分布式 | 功能最全、规模最大 | 部署复杂 |
| Qdrant | 开源 Rust实现 | 性能优秀、易部署 | 生态较小 |
| Weaviate | 开源+SaaS | 多模态原生支持 | 社区活跃度中 |
| Chroma | 开源嵌入式 | 极易上手 | 不适合生产大规模 |
| Pinecone | 闭源SaaS | 免运维 | 成本高、数据出境 |

---

## 6. Milvus 2.5 新特性：稀疏向量与混合检索

### 6.1 稀疏向量（Sparse Vector）原生支持

Milvus 2.5（2024.12）引入原生稀疏向量字段，支持 BM25 等词袋稀疏表示与稠密向量并存于同一 Collection，无需外置 Elasticsearch：

```python
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

connections.connect(host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
]

schema = CollectionSchema(fields, description="混合检索知识库")
collection = Collection("hybrid_kb", schema)

dense_index = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}
collection.create_index("dense_vec", dense_index)

sparse_index = {
    "metric_type": "IP",
    "index_type": "SPARSE_INVERTED_INDEX",
    "params": {"drop_ratio_build": 0.2},
}
collection.create_index("sparse_vec", sparse_index)
collection.load()
```

### 6.2 BM25 稀疏向量构建

使用 `pymilvus` 内置 BM25 函数或手动构建稀疏向量：

```python
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

analyzer = build_default_analyzer(language="zh")
bm25_ef = BM25EmbeddingFunction(analyzer)

corpus = [
    "Tri-Transformer 采用三分支架构检测 AI 幻觉",
    "Milvus 向量数据库支持十亿级 ANN 检索",
    "WebRTC 实现低延迟全双工音视频通信",
]
bm25_ef.fit(corpus)

sparse_embeddings = bm25_ef.encode_documents(corpus)

for i, (doc, sparse_emb) in enumerate(zip(corpus, sparse_embeddings)):
    dense_emb = dense_encoder.encode(doc).tolist()
    collection.insert([{
        "content": doc,
        "dense_vec": dense_emb,
        "sparse_vec": sparse_emb,
    }])
```

### 6.3 混合检索（Hybrid Search）

Milvus 2.5 原生支持单次请求同时执行稠密+稀疏检索，并通过 RRF（Reciprocal Rank Fusion）或 WeightedRanker 融合排序：

```python
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker

def hybrid_search(
    collection: Collection,
    query_text: str,
    dense_encoder,
    bm25_ef,
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list:
    dense_query = dense_encoder.encode(query_text).tolist()
    sparse_query = bm25_ef.encode_queries([query_text])[0]

    dense_req = AnnSearchRequest(
        data=[dense_query],
        anns_field="dense_vec",
        param={"metric_type": "COSINE", "params": {"ef": 100}},
        limit=top_k * 2,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_query],
        anns_field="sparse_vec",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=top_k * 2,
    )

    ranker = WeightedRanker(dense_weight, sparse_weight)

    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=ranker,
        limit=top_k,
        output_fields=["content"],
    )

    return [
        {"content": hit.entity.get("content"), "score": hit.score}
        for hit in results[0]
    ]
```

### 6.4 与 Tri-Transformer RAG 的集成

混合检索显著提升精确关键词命中率，适合技术文档场景（函数名、型号等精确词汇）：

```python
class HybridRAGConnector:
    def __init__(self, collection, dense_encoder, bm25_ef):
        self.collection = collection
        self.dense_encoder = dense_encoder
        self.bm25_ef = bm25_ef

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return hybrid_search(
            self.collection,
            query,
            self.dense_encoder,
            self.bm25_ef,
            top_k=top_k,
        )

    def retrieve_for_planning(
        self, planning_query_tensor: "torch.Tensor", raw_query: str
    ) -> "torch.Tensor":
        import torch
        chunks = self.retrieve(raw_query, top_k=5)
        texts = [c["content"] for c in chunks]
        vecs = self.dense_encoder.encode(texts)
        return torch.tensor(vecs, dtype=torch.float32).unsqueeze(0)
```

### 6.5 Milvus 2.5 其他重要更新

| 特性 | 说明 |
|---|---|
| **JSON 字段索引** | 对 JSON 字段中的嵌套 key 直接建立标量索引，过滤性能提升 10× |
| **Clustering Compaction** | 按标量字段聚类压缩，减少扫描范围，降低查询延迟 |
| **流式写入（Streaming Node）** | 独立 Streaming Node 保证写入后毫秒级可见 |
| **MMap 冷热分层** | 冷数据自动 MMap 到磁盘，热数据保留内存，降低内存成本 50%+ |
