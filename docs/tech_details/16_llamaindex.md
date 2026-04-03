# LlamaIndex（RAG 编排框架）

## 0. 结论先行

- **核心定位**：最受欢迎的开源 RAG 编排框架，提供文档摄入→分块→嵌入→索引→检索→生成完整 Pipeline，与 Milvus/ChromaDB/Qdrant 等向量库原生集成。
- **工程推荐**：直接使用 `llama-index-core` + `llama-index-vector-stores-milvus`；检索策略优先 `HybridRetriever`（向量 + BM25），配合 `SentenceWindowNodeParser`（含前后窗口上下文）；复杂多跳问题用 `SubQuestionQueryEngine`。
- **多模态 RAG**：`MultiModalVectorStoreIndex` 支持图文混合索引，`GPT4VMultiModal` 或本地 Qwen-VL 作为视觉理解节点，文本/图像嵌入分别用 `text-embedding-3-small` / `SigLIP`。
- **Tri-Transformer 中的角色**：知识库构建与检索编排层，负责全流程 RAG 管理；`LlamaAgents`/`Workflows` 可编排 I→C→O 三阶段问答推理；与 Milvus 集成存储多模态嵌入。

---

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

---

## 6. LlamaIndex Workflows 完整实践

### 6.1 事件驱动工作流基础

LlamaIndex Workflows（0.10.x+）基于事件驱动模型，每个 Step 消费一种 Event 并产出下一种 Event，实现复杂多步 RAG 推理：

```python
from llama_index.core.workflow import (
    Workflow, StartEvent, StopEvent, Event, step, Context
)
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from dataclasses import dataclass
from typing import list

@dataclass
class RetrievedEvent(Event):
    nodes: list[NodeWithScore]
    query: str

@dataclass
class RerankEvent(Event):
    nodes: list[NodeWithScore]
    query: str

class TriTransformerRAGWorkflow(Workflow):

    def __init__(self, index: VectorStoreIndex, reranker, llm, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.reranker = reranker
        self.llm = llm

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievedEvent:
        query = ev.query
        retriever = self.index.as_retriever(similarity_top_k=20)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        return RetrievedEvent(nodes=nodes, query=query)

    @step
    async def rerank(self, ctx: Context, ev: RetrievedEvent) -> RerankEvent:
        reranked = self.reranker.postprocess_nodes(ev.nodes, query_str=ev.query)
        return RerankEvent(nodes=reranked[:5], query=ev.query)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        context_str = "\n\n".join([n.node.text for n in ev.nodes])
        prompt = f"上下文:\n{context_str}\n\n问题: {ev.query}\n\n回答:"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))
```

### 6.2 Workflow 执行与流式输出

```python
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank

async def run_rag_workflow():
    llm = OpenAI(model="gpt-4o")
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5
    )

    workflow = TriTransformerRAGWorkflow(
        index=index,
        reranker=reranker,
        llm=llm,
        timeout=60.0,
    )

    result = await workflow.run(query="Tri-Transformer 的幻觉检测机制是什么？")
    print(result)

asyncio.run(run_rag_workflow())
```

---

## 7. LlamaIndex Multi-Agent 协作实践

### 7.1 专业化 Agent 定义

```python
from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool

retrieval_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(similarity_top_k=5),
    name="knowledge_retrieval",
    description="从 Tri-Transformer 技术知识库检索相关文档片段",
)

def check_hallucination(claim: str, evidence: str) -> str:
    score = 0.0
    return f"幻觉评分: {score:.3f}，判断: {'幻觉' if score > 0.5 else '可信'}"

hallucination_tool = FunctionTool.from_defaults(
    fn=check_hallucination,
    name="hallucination_checker",
    description="对给定声明和证据进行幻觉评分",
)

retrieval_agent = FunctionCallingAgent.from_tools(
    tools=[retrieval_tool],
    llm=llm,
    verbose=True,
    system_prompt="你是知识检索专家，负责从知识库找到最相关的技术文档。",
)

verification_agent = FunctionCallingAgent.from_tools(
    tools=[hallucination_tool],
    llm=llm,
    verbose=True,
    system_prompt="你是幻觉检测专家，负责核实信息的准确性。",
)
```

### 7.2 AgentOrchestrator 多 Agent 协作

```python
from llama_index.core.agent.workflow import AgentWorkflow, AgentInput, AgentOutput

async def orchestrate_qa(question: str) -> str:
    retrieval_result = await retrieval_agent.achat(
        f"请检索关于以下问题的相关知识: {question}"
    )
    evidence = str(retrieval_result)

    answer_result = await verification_agent.achat(
        f"问题: {question}\n检索到的证据: {evidence}\n"
        f"请生成回答并检查是否存在幻觉。"
    )

    return str(answer_result)

result = asyncio.run(orchestrate_qa("Tri-Transformer 中 C-Transformer 的作用是什么？"))
print(result)
```

### 7.3 与 Tri-Transformer I→C→O 三阶段对齐

```python
class TriTransformerAgentPipeline:
    """模拟 I→C→O 的 Agent 协作链"""

    def __init__(self, index, llm):
        self.i_agent = FunctionCallingAgent.from_tools(
            tools=[QueryEngineTool.from_defaults(index.as_query_engine())],
            llm=llm,
            system_prompt="负责理解输入并检索相关知识（对应 I-Transformer）",
        )
        self.c_agent = FunctionCallingAgent.from_tools(
            tools=[hallucination_tool],
            llm=llm,
            system_prompt="负责控制与核实，过滤幻觉内容（对应 C-Transformer）",
        )
        self.o_agent = ReActAgent.from_tools(
            tools=[],
            llm=llm,
            system_prompt="负责生成最终高质量回答（对应 O-Transformer）",
        )

    async def run(self, query: str) -> str:
        i_out = await self.i_agent.achat(query)
        c_out = await self.c_agent.achat(
            f"验证以下内容的准确性: {i_out}"
        )
        o_out = await self.o_agent.achat(
            f"基于以下验证过的信息生成完整回答:\n{c_out}\n原始问题: {query}"
        )
        return str(o_out)
```
