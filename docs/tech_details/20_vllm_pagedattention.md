# vLLM & PagedAttention（高效大模型推理服务）

## 0. 结论先行

- **核心贡献**：PagedAttention 受操作系统分页内存启发，将 KV Cache 分块管理（每块 16 Token），消除碎片与预分配浪费，相比 FasterTransformer/Orca 吞吐提升 2-4×，是当前 LLM serving 的事实标准引擎。
- **工程推荐**：`pip install vllm`，5 行代码启动异步推理服务；流式输出用 `AsyncLLMEngine` + WebSocket；量化推理用 `quantization="awq"` 或 `"gptq"`，INT4 量化可再降 40% 显存。
- **关键配置**：`max_num_seqs`（并发请求数）和 `gpu_memory_utilization`（0.85-0.90）是吞吐调优核心参数；多模态推理需 `vllm>=0.5.0` + `--model-type qwen2_vl`。
- **Tri-Transformer 中的角色**：O-Transformer 输出解码的高并发推理引擎；WebSocket/WebRTC 实时流式推理接入层；配合 `prefix_caching=True` 对 RAG 检索内容共享前缀做 KV Cache 复用，降低延迟约 30-50%。

---

## 1. 概述

**vLLM** 是 UC Berkeley 开发的高吞吐量 LLM 推理与服务引擎（arXiv:2309.06180，SOSP 2023 最佳论文），其核心创新 **PagedAttention** 受操作系统虚拟内存与分页技术启发，将 KV Cache 管理精细化，消除内存碎片，并支持 KV Cache 在请求间的灵活共享。相比当时 SOTA 系统（FasterTransformer、Orca），vLLM 在相同延迟下吞吐量提升 **2-4×**。

vLLM 已发展为成熟的多功能推理框架，支持多模态、量化推理、投机解码等特性，是 Tri-Transformer 实时推理服务的首选引擎。

**在 Tri-Transformer 中的角色**：多模态流式推理引擎，支持 WebSocket/WebRTC 推流场景下的高并发低延迟服务；通过 v0.6.0+ 的多步调度和异步输出处理进一步降低延迟。

---

## 2. PagedAttention 原理

### 2.1 KV Cache 的内存问题

自回归生成中，每个 Token 的 KV 向量需要缓存（KV Cache），避免重复计算：

```
13B 模型，seq_len=2048，num_heads=40，head_dim=128，dtype=FP16:
每个请求 KV Cache = 2（K+V）× 40（头）× 128（维）× 2（字节）× 2048（长度）= 40MB

并发 100 请求 = 4GB KV Cache
```

**问题**：传统实现为每个请求预分配**连续的最大长度内存**（如 2048 Token），造成：
- **内部碎片**：请求实际长度 < 最大长度，剩余空间浪费。
- **外部碎片**：可用内存虽足够但不连续，无法分配新请求。

实测：传统系统中 KV Cache 内存利用率仅 **20-40%**。

### 2.2 PagedAttention 解决方案

类比 OS 虚拟内存分页：将 KV Cache 切分为固定大小的**物理块（Block）**，每个 Block 存储固定数量 Token（如 16 Token）的 KV 向量：

```
虚拟块表（Logical Block Table）：
请求 1: [Block0 → 物理页 #7] [Block1 → 物理页 #3] [Block2 → 物理页 #12]
请求 2: [Block0 → 物理页 #1] [Block1 → 物理页 #5]
请求 3: [Block0 → 物理页 #7]  ← 与请求 1 共享！（前缀 KV Cache 复用）

物理 GPU 内存:
[Block #1: req2 B0][Block #3: req1 B1][Block #5: req2 B1]
[Block #7: req1,3 B0（共享）][Block #12: req1 B2]
```

**核心优势**：
1. **近零内碎片**：Block 大小固定，无"最大长度预分配"浪费。
2. **无外碎片**：非连续块通过虚拟块表映射，任意空闲块均可分配。
3. **KV 共享**：相同前缀（如 System Prompt）的 KV 只存一份，多请求共享。

---

## 3. 关键特性

### 3.1 连续批处理（Continuous Batching）

vLLM 不等待整个批次完成，而是当某个请求生成完毕时立即插入新请求：

```
时间步: t0    t1    t2    t3    t4
请求 1: [gen] [gen] [EOS]
请求 2: [gen] [gen] [gen] [EOS]
请求 3:             [插入][gen] [gen]  ← t2 请求1结束后立即插入
请求 4:                   [插入][gen]  ← t3 请求2结束后插入
```

GPU 利用率显著提升，避免等待最慢请求时的空闲。

### 3.2 投机解码（Speculative Decoding）

用小模型（Draft Model）生成多个候选 Token，大模型并行验证：

```python
draft_tokens = small_model.generate(prompt, k=5)
accepted = large_model.verify(prompt, draft_tokens)
```

平均每步大模型调用生成 2-4 个 Token（而非 1 个），延迟降低约 2-3×。

### 3.3 Chunked Prefill

将长 Prompt 的预填充（Prefill）分块执行，避免单个长请求阻塞后续请求的解码，降低首 Token 延迟（TTFT）：

```
传统: [Prefill 4096 tokens] → 等待 ~ 500ms → [Decode]
Chunked: [Prefill chunk1 512] → [decode step] → [Prefill chunk2 512] → ...
```

---

## 4. 使用方法

### 4.1 安装与启动服务

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --port 8000
```

### 4.2 Python 调用

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=32768,
    gpu_memory_utilization=0.9
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "<|endoftext|>"]
)

prompts = ["你好，请介绍一下 Tri-Transformer 架构"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### 4.3 流式生成（适配 WebSocket 推流）

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=32768
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def stream_generate(prompt: str):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    request_id = "req_001"

    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            yield output.outputs[0].text

async def websocket_handler(websocket, path):
    query = await websocket.recv()
    async for token in stream_generate(query):
        await websocket.send(token)
```

### 4.4 多模态支持（vLLM v0.6+）

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    dtype="bfloat16",
    max_model_len=32768
)

image = Image.open("image.jpg")
prompt = "<|vision_start|><|image_pad|><|vision_end|>描述这张图片"

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}, SamplingParams(max_tokens=256))
```

---

## 5. 性能调优

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| `gpu_memory_utilization` | 0.85-0.92 | KV Cache 占 GPU 内存比例 |
| `max_num_seqs` | 128-512 | 最大并发请求数 |
| `tensor_parallel_size` | GPU 数 | 多卡张量并行 |
| `block_size` | 16 | PagedAttention 块大小（Token 数）|
| `max_model_len` | 按需 | 越小 KV Cache 越省内存 |

---

## 6. 最新进展（2024-2025）

### 6.1 vLLM v0.6.0（2024 SOSP）
- **2.7× 吞吐量提升**（Llama 8B）：进程分离（ZMQ）消除 Python GIL 瓶颈、多步调度、异步输出处理。
- 超越 TensorRT-LLM、SGLang 成为 H100 上的 SOTA 推理引擎（ShareGPT 基准）。

### 6.2 SGLang（2024）
- 专为复杂推理程序（多轮对话、树形搜索）优化的推理引擎，RadixAttention 实现更激进的 KV Cache 共享，在某些场景超越 vLLM。

### 6.3 MLA（Multi-head Latent Attention，DeepSeek-V2）
- DeepSeek-V2 提出的 KV Cache 压缩技术，通过低秩 KV 投影将 KV Cache 压缩 5-10×，配合 vLLM 使用可大幅提升并发能力。

### 6.4 与 Tri-Transformer 的集成
- Tri-Transformer 的三分支推理可在 vLLM 框架上实现为**自定义模型类（Custom Model）**，利用 vLLM 的 Continuous Batching + PagedAttention 服务多模态流式请求。
