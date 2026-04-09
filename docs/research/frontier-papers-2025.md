# 前沿论文与技术进展综述（2024-2026）

**面向 Tri-Transformer 项目的研究雷达报告**

**版本**: v1.0  
**更新日期**: 2026-04-13  
**覆盖方向**: Transformer 架构创新 / 幻觉检测与缓解 / 多模态大模型 / 实时语音模型 / DiT 生成模型 / LoRA/PEFT 高效微调 / 推理加速 / RAG 进展

---

## 目录

1. [方向一：Transformer 架构创新](#1-transformer-架构创新)
2. [方向二：幻觉检测与缓解](#2-幻觉检测与缓解)
3. [方向三：多模态大模型](#3-多模态大模型)
4. [方向四：实时语音模型](#4-实时语音模型)
5. [方向五：DiT 生成模型](#5-dit-生成模型)
6. [方向六：LoRA/PEFT 高效微调](#6-lorapeft-高效微调)
7. [方向七：推理加速](#7-推理加速)
8. [方向八：RAG 进展](#8-rag-进展)
9. [优先级行动建议](#9-优先级行动建议)
10. [Tri-Transformer 演进路线图](#10-tri-transformer-演进路线图)

---

## 1. Transformer 架构创新

### 1.1 Mamba-2 / State Space Duality (SSD)

| 属性 | 详情 |
|------|------|
| **标题** | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |
| **来源** | arXiv:2405.21060 \| ICML 2024 |
| **作者** | Tri Dao, Albert Gu |
| **时间** | 2024 年 5 月 |

**核心贡献**：

建立 Transformer 与 SSM（状态空间模型）之间的数学等价关系，通过"结构化半可分矩阵"框架统一两类架构。提出 Mamba-2，硬件友好，训练速度比 Mamba-1 快 2-8 倍，在语言建模上与 Transformer 竞争。揭示选择性 SSM 本质上是一种受约束的注意力变体。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 三分支架构（ITransformer / CTransformer / OTransformer）若需降低 KV-Cache 显存，可在 I 或 O 分支中将部分 Decoder Layer 替换为 SSD 层，实现线性复杂度推理同时保持表达能力。MLA（见条目 16）与 SSD 框架可联合使用以压缩显存占用。

---

### 1.2 xLSTM: Extended Long Short-Term Memory

| 属性 | 详情 |
|------|------|
| **标题** | xLSTM: Extended Long Short-Term Memory |
| **来源** | arXiv:2405.04517 |
| **时间** | 2024 年 5 月（2024 年 12 月修订） |

**核心贡献**：

引入指数门控（exponential gating）+ 归一化稳定化技术，突破传统 LSTM 的记忆瓶颈。提出 sLSTM（标量记忆、内存混合）和 mLSTM（矩阵记忆、协方差更新规则，完全并行化）两种结构。在性能和扩展性上与 Transformer 及 SSM 模型持平，在某些任务上超越 Mamba。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

xLSTM 的矩阵记忆机制（mLSTM）与 CTransformer 的 State Slots（16 个状态插槽 + Cross-Attn）存在设计相似性；mLSTM 可作为 C-Transformer 控制中枢的候选内部记忆模块，提升长上下文的状态保留能力。

---

### 1.3 TTT: Test-Time Training as a Hidden-State Update Rule

| 属性 | 详情 |
|------|------|
| **标题** | Learning to (Learn at Test Time): RNNs with Expressive Hidden States |
| **来源** | arXiv:2407.04620 |
| **时间** | 2024 年 7 月（2025 年 8 月更新至 v4） |

**核心贡献**：

将隐状态本身设计为一个可在测试时更新的机器学习模型（线性模型或 MLP），更新规则即自监督学习步骤。提出 TTT-Linear 和 TTT-MLP 两种实例，实现线性时间复杂度。与 Mamba 相比，TTT 层在超过 16k Token 的长上下文下仍持续降低困惑度（Mamba 在此处饱和），表现与 Transformer 接近。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 的 ITransformer 采用 Qwen3 骨干（36 层），在长文档 RAG 场景下对 >16k Token 的上下文建模是核心挑战。TTT 层可作为对角 Attention 的替代，嵌入 I-Transformer 的 Decoder Layer 中，以更低显存代价处理长上下文 RAG 检索。

---

### 1.4 次二次复杂度架构综合调研

| 属性 | 详情 |
|------|------|
| **标题** | The End of Transformers? On Challenging Attention and the Sub-Quadratic Frontier |
| **来源** | arXiv:2510.05364 |
| **时间** | 2025 年 10 月 |

**核心贡献**：

系统性综述 Mamba、RWKV、xLSTM、GLA、TTT 等次二次复杂度架构的能力边界与局限。评估这些架构在长序列、in-context learning 和各类下游任务上与纯注意力 Transformer 的差距。明确指出：现有替代架构在 recall-intensive（需精确检索）任务上仍与 Transformer 有显著差距。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

本 Survey 对 Tri-Transformer 选择注意力骨干（Qwen3 GQA）而非 SSM 架构提供了理论支撑；同时指出混合架构（Hybrid Attention + SSM）是未来的优化方向，可指导 Tri-Transformer 后续架构迭代。

---

## 2. 幻觉检测与缓解

### 2.1 LLM 幻觉综合调研（2025 更新版）

| 属性 | 详情 |
|------|------|
| **标题** | Large Language Models Hallucination: A Comprehensive Survey |
| **来源** | arXiv:2510.06265 |
| **时间** | 2025 年 10 月（v3 更新至 2026 年 3 月） |

**核心贡献**：

从 LLM 全开发生命周期（数据、架构设计、推理）系统梳理幻觉的根本原因。建立幻觉检测方法分类体系：

- 基于内部表征的方法（hidden states、attention map）
- 基于输出一致性的方法（self-consistency）
- 基于外部知识的方法（RAG + KG）

整理当前最全的幻觉基准测试和评估指标综述。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

直接对应 Tri-Transformer 核心任务。`eval/loss/hallucination_loss.py` 和 `HallucinationEvaluator` 的设计可参考本 Survey 中的多维度检测分类体系；三分支架构（ITransformer 负责内部一致性、CTransformer 负责控制对齐、OTransformer 负责输出验证）天然对应 Survey 中提出的三种检测维度。

---

### 2.2 ProgRAG：渐进式知识图谱检索抗幻觉

| 属性 | 详情 |
|------|------|
| **标题** | ProgRAG: Hallucination-Resistant Progressive Retrieval and Generation over Knowledge Graphs |
| **来源** | arXiv:2511.10240 |
| **时间** | 2025 年 11 月 |

**核心贡献**：

面向复杂多跳知识图谱问答，将问题分解为子问题，逐步扩展推理路径。引入不确定性感知剪枝（uncertainty-aware pruning）机制，通过 LLM 筛选低质候选证据。通过部分路径优化推理上下文，显著减少 KG 检索引入的幻觉。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 的 RAG 管线（`RAGRetriever` + ChromaDB/Milvus）目前缺乏多跳推理能力；ProgRAG 的渐进式分解策略可直接应用于 `app/services/chat/` 的检索逻辑，与 ITransformer 的长程建模配合实现更复杂的知识推理链。

---

### 2.3 CogGRAG：仿人类认知图谱推理

| 属性 | 详情 |
|------|------|
| **标题** | Human Cognition Inspired RAG with Knowledge Graph for Complex Reasoning |
| **来源** | arXiv:2503.06567 \| AAAI 2026 |
| **时间** | 2025 年 3 月 |

**核心贡献**：

提出 CogGRAG，将推理过程建模为树形思维导图，自顶向下分解问题，自底向上综合知识图谱证据。三阶段 Pipeline：问题分解 → 结构化 KG 检索 → 双过程自验证（dual-process self-verification）。统一了问题分解、知识检索和推理为单一图结构认知框架，超越 MindMap、Graph-CoT 等前作。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

双过程自验证机制与 Tri-Transformer 的三分支设计高度契合：ITransformer 负责输入事实核查，CTransformer 作为控制中枢（类比 CogGRAG 的树形分解器），OTransformer 负责输出规划验证。可将 CogGRAG 框架作为 `FactChecker` 服务的升级方案。

---

## 3. 多模态大模型

### 3.1 Qwen2.5-VL 技术报告

| 属性 | 详情 |
|------|------|
| **标题** | Qwen2.5-VL Technical Report |
| **来源** | arXiv:2502.13923 |
| **时间** | 2025 年 2 月 |

**核心贡献**：

从零训练动态分辨率 ViT（Native Dynamic-Resolution ViT）+ Window Attention，支持任意图像尺寸输入。引入绝对时间编码（Absolute Time Encoding）用于视频的秒级事件定位，可处理数小时长视频。72B 旗舰模型在文档理解、图表分析、代理操作等任务上匹配 GPT-4o 和 Claude 3.5 Sonnet。具备 OCR、结构化数据抽取（发票、表格）、跨语言多模态理解等企业级能力。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

Tri-Transformer 的 ITransformer 当前采用 Qwen3 纯文本骨干；若需扩展为多模态输入（如上传文档、图像检索），Qwen2.5-VL 的动态分辨率 ViT 可作为 I-Transformer 的视觉编码前置模块，无缝对接现有 4096 维 hidden 维度。

---

### 3.2 InternVL3：开源多模态大模型新 SOTA

| 属性 | 详情 |
|------|------|
| **标题** | InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models |
| **来源** | arXiv:2504.10479 |
| **时间** | 2025 年 4 月 |

**核心贡献**：

原生多模态预训练：首次实现多模态数据与纯文本数据在单一预训练阶段联合学习，消除模态分离训练的对齐代价。引入 V2PE（Variable Visual Position Encoding）支持超长多模态上下文。结合 SFT、MPO（混合偏好优化）和测试时扩展策略，InternVL3-78B 在 MMMU 基准上达 72.2 分，创开源模型新高。开源权重和训练数据，匹配 ChatGPT-4o、Claude 3.5 Sonnet、Gemini 2.5 Pro 的能力。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

InternVL3 的原生多模态预训练范式和 V2PE 对 Tri-Transformer 的扩展路径（多模态幻觉检测）具有重要参考价值；MPO 训练策略可用于 Tri-Transformer 的对齐阶段（`control_alignment_loss.py`）。

---

## 4. 实时语音模型

### 4.1 Moshi：首个全双工实时语音对话模型

| 属性 | 详情 |
|------|------|
| **标题** | Moshi: A Speech-Text Foundation Model for Real-Time Dialogue |
| **来源** | arXiv:2410.00037 \| Kyutai Labs |
| **时间** | 2024 年 9 月（已开源） |

**核心贡献**：

首个全双工（Full-Duplex）实时语音大语言模型，能同时处理用户输入和自身生成，消除传统"轮流说话"限制。提出 Inner Monologue 内心独白机制：在生成音频 Token 前预测时间对齐的文本 Token，显著提升语言质量。理论延迟 160ms，实际延迟 200ms；将语音对话建模为语音到语音生成任务。采用 Mimi 音频分词器，蒸馏语义信息到第一层声学 Token。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 具备 WebSocket 实时通信基础设施（`/api/v1/model/stream`），结合 Moshi 的 Inner Monologue 思想，可在 OTransformer 的 Streaming Decoder 中实现"先生成内部文字规划 Token，再生成最终输出 Token"的双流生成机制，大幅提升实时对话的语言一致性和幻觉控制能力。

---

### 4.2 VITA-1.5：GPT-4o 级别实时视觉与语音交互

| 属性 | 详情 |
|------|------|
| **标题** | VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction |
| **来源** | arXiv:2501.01957 \| NeurIPS 2025 Spotlight |
| **时间** | 2025 年 1 月 |

**核心贡献**：

多阶段渐进式训练方法，使 LLM 同时理解视觉和语音信息，无需独立的 ASR 和 TTS 模块。端到端 Speech-to-Speech 对话能力，保留强视觉-语言性能的同时实现低延迟响应。开源实现（2.4K GitHub Stars），在 NeurIPS 2025 中作为 Spotlight 展示。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

VITA-1.5 的无 ASR/TTS 端到端架构是 Tri-Transformer 未来集成语音通道的重要参考；其多阶段训练方法可直接用于 Tri-Transformer 的多模态扩展训练流程（对应 `app/services/train/TrainingService`）。

---

## 5. DiT 生成模型

### 5.1 DiTCtrl：MM-DiT 多提示视频生成

| 属性 | 详情 |
|------|------|
| **标题** | DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Zero-Shot Multi-Prompt Video Generation |
| **来源** | arXiv:2412.18597 \| CVPR 2025 |
| **时间** | 2024 年 12 月 |

**核心贡献**：

首个针对 MM-DiT（多模态扩散 Transformer）架构的无训练多提示视频生成方法。通过分析 MM-DiT 的 3D 全注意力机制，发现其与 UNet 中 Cross/Self-Attention 行为等价，实现基于掩码的语义控制和注意力共享。同步提出 MPVBench 多提示视频评测基准。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

Tri-Transformer 的 CTransformer 采用 DiT 控制中枢设计（adaLN-Zero + State Slots）；DiTCtrl 揭示的 3D 全注意力控制机制对 CTransformer 如何利用注意力图对 I/O 分支进行跨模态控制具有直接借鉴价值，可改进 `eval/loss/control_alignment_loss.py` 的对齐信号设计。

---

## 6. LoRA/PEFT 高效微调

### 6.1 DoRA：分解幅度与方向的权重微调

| 属性 | 详情 |
|------|------|
| **标题** | DoRA: Weight-Decomposed Low-Rank Adaptation |
| **来源** | arXiv:2402.09353 \| ICML 2024 Oral |
| **时间** | 2024 年 2 月 |

**核心贡献**：

将预训练权重分解为幅度（magnitude）和方向（direction）两个分量，分别微调。LoRA 仅用于方向分量的更新，消除全量微调与 LoRA 之间的行为差异。在 LLaMA、LLaVA、VL-BART 等多种下游任务上一致超越 LoRA，且无额外推理开销。NVlabs 官方开源（github.com/NVlabs/DoRA）。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 已实现 LoRA 适配器（`app/model/lora_adapter.py`），可直接升级为 DoRA：将权重分解逻辑加入适配器层，尤其在 ITransformer 和 OTransformer 骨干（Qwen3-8B）的微调中，DoRA 的方向-幅度分解能更好保留骨干的知识，提升幻觉检测任务的精度。

---

### 6.2 GaLore：全参数梯度低秩投影训练

| 属性 | 详情 |
|------|------|
| **标题** | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection |
| **来源** | arXiv:2403.03507 \| ICML 2024 Oral |
| **时间** | 2024 年 3 月 |

**核心贡献**：

对梯度（而非权重）进行低秩投影，实现全参数学习同时降低优化器状态显存。减少优化器状态显存高达 65.5%（8-bit GaLore 达 82.5%）。首次在单张 RTX 4090（24GB）上成功预训练 7B 规模模型，无需并行、梯度检查点或卸载。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 的三分支架构（两个 Qwen3-8B + DiT 控制层，总显存约 33GB）在单卡训练时面临严重的显存压力。GaLore 可集成到 `TrainingService` 的优化器配置中，使在消费级 GPU 上进行全量参数微调成为可能，对 Tri-Transformer 的持续训练（continual training）具有重要意义。

---

### 6.3 LoRA+：不同学习率的矩阵分量

| 属性 | 详情 |
|------|------|
| **标题** | LoRA+: Efficient Low Rank Adaptation of Large Models |
| **来源** | arXiv:2402.12354 |
| **时间** | 2024 年 2 月 |

**核心贡献**：

发现原始 LoRA 对 A、B 两个矩阵使用相同学习率在宽模型中阻碍高效特征学习。提出对 A 和 B 矩阵使用不同学习率比例，在不增加计算开销的前提下实现 1-2% 性能提升和最高 2 倍微调加速。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

可直接应用于 `lora_adapter.py`，通过调整 A/B 矩阵的学习率比值（推荐比值在 4-16 之间），加速 Tri-Transformer 的 LoRA 微调收敛速度，实验成本极低。

---

## 7. 推理加速

### 7.1 DeepSeek-V3：MLA + MoE 高效推理架构

| 属性 | 详情 |
|------|------|
| **标题** | DeepSeek-V3 Technical Report |
| **来源** | arXiv:2412.19437 \| DeepSeek AI |
| **时间** | 2024 年 12 月（2025 年 2 月更新） |

**核心贡献**：

**MLA（Multi-Head Latent Attention）**：通过低秩压缩将 KV 缓存压缩到潜在向量，大幅降低推理时显存占用，同时保持全注意力表达能力。**DeepSeekMoE**：671B 总参数，每 Token 仅激活 37B，实现极致推理效率。首创无辅助损失负载均衡（Auxiliary-Loss-Free Load Balancing），消除传统 MoE 的路由损失干扰。**多 Token 预测**（Multi-Token Prediction）训练目标，提升推理速度和性能。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

MLA 的 KV 缓存压缩技术可直接应用于 Tri-Transformer 的 ITransformer 和 OTransformer（Qwen3-8B GQA 骨干），进一步降低 16GB/分支 的显存占用；结合 Qwen3-30B-A3B MoE 配置，Tri-Transformer 的 MoE 方案可参考 DeepSeek-V3 的无辅助损失路由，避免专家坍塌问题。

---

## 8. RAG 进展

### 8.1 GraphRAG：大规模私有语料图谱问答

| 属性 | 详情 |
|------|------|
| **标题** | From Local to Global: A Graph RAG Approach to Query-Focused Summarization |
| **来源** | arXiv:2404.16130 \| Microsoft Research |
| **时间** | 2024 年 4 月（2025 年 2 月更新） |

**核心贡献**：

指出传统 RAG 在全局性问题（需理解整个语料库的问题）上完全失效。GraphRAG 两阶段索引：LLM 构建实体知识图谱 → 生成社区摘要（community summaries）。分层答案合成：利用社区摘要生成局部回答，再汇总为全局答案。在 1M Token 规模语料上，GraphRAG 在答案全面性和多样性上大幅超越传统 RAG 基线。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

Tri-Transformer 当前 RAG 管线基于 ChromaDB/Milvus 向量检索（`RAGRetriever`），缺乏全局语义理解能力。GraphRAG 的社区摘要机制可作为知识库索引的补充层，解决多文档跨篇章推理问题，直接提升 `rag_loss.py` 评测中的召回精度。

---

### 8.2 HippoRAG：仿海马体长期记忆 RAG

| 属性 | 详情 |
|------|------|
| **标题** | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models |
| **来源** | arXiv:2405.14831 \| NeurIPS 2024 |
| **时间** | 2024 年 5 月 |

**核心贡献**：

受海马体索引理论启发，协同编排 LLM、知识图谱和 Personalized PageRank 算法。在多跳问答上超越 SOTA 方法最多 20%。单步检索达到迭代检索（IRCoT）的同等或更好效果，同时速度提升 6-13 倍，成本降低 10-30 倍。能处理现有方法无法解决的新型检索场景。

**与 Tri-Transformer 关联** — 关联等级：A（高度相关）

HippoRAG 的神经生物学启发设计与 Tri-Transformer 的三分支拟脑架构（ITransformer ↔ 感知层 / CTransformer ↔ 控制层 / OTransformer ↔ 输出层）存在深层对应关系；其 Personalized PageRank 检索机制可替换或增强 `RAGRetriever` 的 BM25 重排序策略，尤其在知识密集型幻觉检测场景中效果显著。

---

### 8.3 RAPTOR：树形分层 RAG

| 属性 | 详情 |
|------|------|
| **标题** | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval |
| **来源** | arXiv:2401.18059 \| ICLR 2024 |
| **时间** | 2024 年 1 月 |

**核心贡献**：

对文本递归嵌入、聚类和摘要，自底向上构建多级摘要树。检索时跨不同抽象层次整合信息，支持复杂多步推理。结合 GPT-4 在 QuALITY 基准上绝对准确率提升 20%。

**与 Tri-Transformer 关联** — 关联等级：B（中度相关）

RAPTOR 树形索引与 HippoRAG 形成互补；RAPTOR 擅长长文档摘要层次检索，HippoRAG 擅长多跳事实精确检索。两者结合可构建 Tri-Transformer 更完整的知识库检索架构。

---

## 9. 优先级行动建议

基于上述 18 篇论文的综合分析，按 Tri-Transformer 各模块列出优先级行动建议：

### P0 — 立即可落地（成本低、收益高）

| 编号 | 论文 | 目标模块 | 建议行动 |
|------|------|---------|---------|
| A1 | DoRA (arXiv:2402.09353) | `lora_adapter.py` | 将 LoRA 适配器升级为 DoRA，提升微调精度 |
| A2 | LoRA+ (arXiv:2402.12354) | `lora_adapter.py` | 为 A、B 矩阵设置不同学习率（比值 4-16），加速收敛 |
| A3 | GaLore (arXiv:2403.03507) | `TrainingService` | 集成 GaLore 优化器，降低单卡显存 65%+ |
| A4 | HippoRAG (arXiv:2405.14831) | `RAGRetriever` | 引入 Personalized PageRank 替换 BM25 重排序 |
| A5 | GraphRAG (arXiv:2404.16130) | `RAGRetriever` | 为知识库添加社区摘要全局索引层 |

### P1 — 中期迭代（Phase 2 音频模态）

| 编号 | 论文 | 目标模块 | 建议行动 |
|------|------|---------|---------|
| B1 | Moshi Inner Monologue (arXiv:2410.00037) | `OTransformer` | 实现双流生成（文本规划 Token → 音频 Token） |
| B2 | CogGRAG (arXiv:2503.06567) | `FactChecker` | 引入双过程自验证机制升级 FactChecker 服务 |
| B3 | ProgRAG (arXiv:2511.10240) | `ChatService` | 加入多跳渐进式 KG 检索推理 |
| B4 | DeepSeek MLA (arXiv:2412.19437) | `ITransformer/OTransformer` | 实现 KV Cache 低秩压缩，降低 I/O 分支推理显存 |
| B5 | TTT (arXiv:2407.04620) | `ITransformer` | 超长上下文（>16k）场景引入 TTT 层替代部分 Attention |

### P2 — 长期规划（Phase 3 全模态）

| 编号 | 论文 | 目标模块 | 建议行动 |
|------|------|---------|---------|
| C1 | Qwen2.5-VL (arXiv:2502.13923) | `ITransformer` | Phase 3 视觉编码前置模块 |
| C2 | InternVL3 (arXiv:2504.10479) | 训练 Pipeline | 参考原生多模态预训练范式 |
| C3 | VITA-1.5 (arXiv:2501.01957) | 整体架构 | Phase 3 无 ASR/TTS 端到端语音-视觉方案 |
| C4 | DiTCtrl (arXiv:2412.18597) | `CTransformer` | 改进 adaLN-Zero 注意力控制机制 |
| C5 | Mamba-2 SSD (arXiv:2405.21060) | `ITransformer/OTransformer` | 混合 Attention + SSD 架构降低推理复杂度 |

---

## 10. Tri-Transformer 演进路线图

基于最新研究进展，建议 Tri-Transformer 按以下路线演进：

```
当前状态（v2.0，已完成）
├── Phase 1 MVP 文本模态
│   ├── ✅ PyTorch Tri-Transformer 骨架（I/C/O 三分支）
│   ├── ✅ Qwen3 骨干插拔
│   ├── ✅ 文本 BPE Tokenizer
│   ├── ✅ FastAPI 后端 + React 前端
│   ├── ✅ ChromaDB + BM25 文本 RAG
│   └── ✅ 自定义损失函数 + 评估管道
│
近期优化（基于前沿论文，低成本高收益）
├── 🔧 DoRA + LoRA+ 替换现有 LoRA 适配器（P0，~1 天）
├── 🔧 GaLore 集成优化器（P0，~2 天）
├── 🔧 HippoRAG 检索升级（P0，~3 天）
└── 🔧 GraphRAG 全局索引补充（P0，~3 天）
│
Phase 2（2026 Q2-Q3，Audio-Text 语音交互）
├── Inner Monologue 双流生成机制（Moshi 方案）
├── SNAC 音频 Tokenizer + WebRTC 全双工
├── CogGRAG 双过程自验证 FactChecker
├── ProgRAG 多跳推理 RAG 管线
├── DeepSeek MLA KV Cache 压缩
└── TTT 超长上下文处理层
│
Phase 3（2026 Q4+，Omni-Modal 全模态）
├── Qwen2.5-VL / InternVL3 视觉编码器接入
├── VITA-1.5 端到端无 ASR/TTS 架构
├── Mamba-2 混合注意力架构
├── DiTCtrl 注意力控制机制优化
└── Milvus 多模态向量库（图像/音频/文本混合检索）
```

### 技术优先级矩阵

```
高影响力 │ GraphRAG    HippoRAG   │ MLA压缩    TTT
         │ DoRA/GaLore ProgRAG    │ Inner Mono CogGRAG
─────────┼───────────────────────┼──────────────────
低影响力 │ LoRA+       DiTCtrl   │ Qwen2.5-VL VITA-1.5
         │ xLSTM       RAPTOR    │ Mamba-2    InternVL3
         └──────────────────────┴──────────────────
               低实现成本              高实现成本
```

---

## 参考文献索引

| 编号 | arXiv ID | 简称 | 方向 |
|------|----------|------|------|
| 1 | arXiv:2405.21060 | Mamba-2 / SSD | 架构 |
| 2 | arXiv:2405.04517 | xLSTM | 架构 |
| 3 | arXiv:2407.04620 | TTT | 架构 |
| 4 | arXiv:2510.05364 | 次二次架构 Survey | 架构 |
| 5 | arXiv:2510.06265 | 幻觉 Survey 2025 | 幻觉 |
| 6 | arXiv:2511.10240 | ProgRAG | 幻觉/RAG |
| 7 | arXiv:2503.06567 | CogGRAG | 幻觉/KG |
| 8 | arXiv:2502.13923 | Qwen2.5-VL | 多模态 |
| 9 | arXiv:2504.10479 | InternVL3 | 多模态 |
| 10 | arXiv:2410.00037 | Moshi | 语音 |
| 11 | arXiv:2501.01957 | VITA-1.5 | 语音 |
| 12 | arXiv:2412.18597 | DiTCtrl | DiT |
| 13 | arXiv:2402.09353 | DoRA | PEFT |
| 14 | arXiv:2403.03507 | GaLore | PEFT |
| 15 | arXiv:2402.12354 | LoRA+ | PEFT |
| 16 | arXiv:2412.19437 | DeepSeek-V3 MLA | 推理 |
| 17 | arXiv:2404.16130 | GraphRAG | RAG |
| 18 | arXiv:2405.14831 | HippoRAG | RAG |
| 19 | arXiv:2401.18059 | RAPTOR | RAG |
