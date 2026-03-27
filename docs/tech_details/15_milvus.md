# Milvus（多模态向量数据库）

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
