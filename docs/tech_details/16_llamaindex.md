# LlamaIndex（RAG 编排框架）

## 1. 概述

LlamaIndex（原名 GPT Index）是由 Jerry Liu 等人创建的开源数据框架，专为将外部数据与 LLM 应用集成而设计，是目前最受欢迎的 RAG 编排框架之一（GitHub 48.1K stars）。LlamaIndex 提供从文档摄入（Ingestion）、索引构建（Indexing）、检索（Retrieval）到生成（Generation）的完整 RAG Pipeline，并原生支持多模态、Agent、工作流等高级场景。

**在 Tri-Transformer 中的角色**：知识库构建与检索编排层，负责文档解析、分块、嵌入、索引和查询的全流程管理，与 Milvus 向量数据库原生集成。

---

## 2. 核心架构

### 2.1 LlamaIndex 核心抽象层

```
┌──────────────────────────────────────────────────────┐
│                   LlamaIndex Stack                    │
├──────────────────────────────────────────────────────┤
│  QueryEngine / AgentRunner  （查询/代理入口）           │
├──────────────────────────────────────────────────────┤
│  Synthesizer  （答案合成）   Retriever（检索器）        │
├──────────────────────────────────────────────────────┤
│  VectorStoreIndex / KnowledgeGraphIndex / ... （索引） │
├──────────────────────────────────────────────────────┤
│  NodeParser（分块） + Embedder（向量化）                │
├──────────────────────────────────────────────────────┤
│  DocumentLoader（文档加载）：PDF/Word/Web/Audio/Video  │
├──────────────────────────────────────────────────────┤
│  LLM Backend + Embedding Model Backend                │
└──────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
原始文档（PDF/音视频/代码/网页）
    ↓ [DocumentLoader] 加载
Document 对象（含元数据）
    ↓ [NodeParser] 分块
Node 对象列表（每个 Node = 一个文本块）
    ↓ [Embedder] 向量化
(Node, Embedding) 对
    ↓ [VectorStore] 存储（Milvus/Chroma/Pinecone）
索引构建完成
    --------- 查询阶段 ---------
用户查询
    ↓ [QueryTransformer] 查询改写（可选）
    ↓ [Retriever] 检索 Top-K Nodes
    ↓ [NodePostprocessors] 重排序/压缩/过滤
    ↓ [ResponseSynthesizer] 合成答案
最终回答
```

---

## 3. 使用方法

### 3.1 安装

```bash
pip install llama-index llama-index-vector-stores-milvus
pip install llama-index-embeddings-huggingface
pip install llama-index-readers-file
```

### 3.2 基础 RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)
Settings.llm = OpenAI(model="gpt-4o")

documents = SimpleDirectoryReader("./knowledge_base/").load_data()

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist("./index_store")

query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("Tri-Transformer 的架构是什么？")
print(response)
```

### 3.3 与 Milvus 集成

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="tri_transformer_kb",
    dim=1024,
    overwrite=False
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)
```

### 3.4 高级查询：混合检索 + 重排序

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import SimilarityPostprocessor

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20,
)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5
)

similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.7)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[similarity_filter, reranker],
)

response = query_engine.query("RAG 的幻觉阻断机制如何实现？")
for node in response.source_nodes:
    print(f"[{node.score:.3f}] {node.node.text[:100]}...")
```

### 3.5 多模态 RAG

```python
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    "./multimodal_kb/",
    required_exts=[".jpg", ".png", ".pdf", ".txt", ".mp4"]
).load_data()

mm_index = MultiModalVectorStoreIndex.from_documents(documents)

mm_query_engine = mm_index.as_query_engine(
    multi_modal_llm=OpenAIMultiModal(model="gpt-4o"),
    similarity_top_k=3,
)

response = mm_query_engine.query(
    "根据知识库中的图表，说明产品的安装步骤",
    image_documents=[]
)
```

### 3.6 自定义文档解析器

```python
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document

semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model
)

raw_text = open("technical_doc.txt").read()
document = Document(text=raw_text, metadata={"source": "technical_doc.txt"})
nodes = semantic_splitter.get_nodes_from_documents([document])
print(f"语义分块数量: {len(nodes)}")
```

---

## 4. LlamaParse（高级文档解析）

LlamaIndex 提供 LlamaParse 服务，支持复杂 PDF（含表格、图表、公式）的高质量解析：

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="your_api_key",
    result_type="markdown",
    language="ch_sim",
    parsing_instruction="提取所有表格和数字数据"
)

documents = parser.load_data("technical_report.pdf")
```

---

## 5. 最新进展（2024-2025）

### 5.1 LlamaIndex Workflows（2024）
- 新一代事件驱动的 Agent 工作流系统，支持复杂多步 RAG 推理（如：先检索文档目录 → 再检索具体章节 → 再核实数据来源）。

### 5.2 LlamaAgents（2024）
- 多 Agent 协作框架，多个专业 Agent（文档检索 Agent、代码执行 Agent、数学计算 Agent）协同完成复杂任务。

### 5.3 Structured Outputs
- 通过 Pydantic 模型定义输出 Schema，LLM 生成的答案自动解析为结构化数据，适合 Tri-Transformer 中的知识核实场景。

### 5.4 与 Tri-Transformer 的集成建议
- **知识摄入阶段**：LlamaParse 解析 PDF 技术文档 → SemanticSplitter 语义分块 → BGE-M3 嵌入 → Milvus 存储。
- **实时查询阶段**：LlamaIndex Retriever 接收 O-Transformer 编码的查询向量 → 混合检索 Top-5 → Cross-Encoder 重排序 → 注入 Planning Encoder。
