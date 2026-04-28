# 产品需求文档（PRD）：Tri-Transformer 实时多模态可控对话与 RAG 知识库增强系统

**版本**：v2.1  
**更新日期**：2026-04-28  
**状态**：研究进展驱动完善版 + Jetson Nano 边缘部署

---

## 目录

1. [产品概述](#1-产品概述)
2. [需求背景与研究依据](#2-需求背景与研究依据)
3. [系统架构总体设计](#3-系统架构总体设计)
4. [核心模块功能需求](#4-核心模块功能需求)
   - 4.1 I-Transformer：正向实时输入编码
   - 4.2 C-Transformer：DiT 生成控制中枢
   - 4.3 O-Transformer：反向实时输出解码
   - 4.4 多模态统一 Tokenizer
   - 4.5 RAG 知识库与幻觉控制
   - 4.6 联合训练 Pipeline
   - 4.7 推理与部署
   - 4.7.2 Jetson Nano 边缘部署方案
5. [前端产品功能需求](#5-前端产品功能需求)
6. [后端 API 需求](#6-后端-api-需求)
7. [评估体系需求](#7-评估体系需求)
8. [非功能性需求](#8-非功能性需求)
9. [技术栈选型](#9-技术栈选型)
10. [阶段规划与里程碑](#10-阶段规划与里程碑)
11. [验收标准](#11-验收标准)

---

## 1. 产品概述

### 1.1 产品名称

**Tri-Transformer 实时多模态可控对话与 RAG 知识库增强系统**（简称：Tri-Transformer）

### 1.2 产品定位

一款融合**三分支 Transformer 深度扭合创新架构**与**原生实时连续多模态交互**、**向量化文档知识库（RAG）** 的高端 AI 系统。系统创新性地将正向 Decoder-Encoder（I-Transformer）、DiT 控制中枢（C-Transformer）、反向 Encoder-Decoder（O-Transformer）三个功能异构的 Transformer 模块扭合为一个端到端可训练的可微整体，实现全双工（Full-duplex）音视频/文本连续输入输出，并提供高可控、无幻觉的领域级知识生成。

### 1.3 核心创新点

| 创新维度 | 描述 |
|---|---|
| **三分支扭合架构** | I（Dec→Enc）→ C（DiT）→ O（Enc→Dec）闭环，打破单向计算图，实现中途硬性可控 |
| **双端异构插拔** | 训练/推理时 I/O 两端可独立插接不同开源大模型（Qwen3-8B/14B/30B-MoE 等） |
| **实时多模态流** | 音频/视频/文本 Token 统一离散化，支持 50ms 级切片流式处理 |
| **无幻觉阻断** | C-Transformer 实时监控 I 端编码与 O 端规划的语义一致性，不一致时强制干预 |
| **全双工对话** | 用户打断即响应（< 200ms），类 Moshi 的连续双向音频流 |

### 1.4 目标用户

| 用户类型 | 场景 |
|---|---|
| AI 架构师 / 算法工程师 | 研究三分支扭合架构，插拔不同骨干，进行多模态对齐实验 |
| 企业知识管理员 | 构建私有多模态知识库，实现无幻觉领域问答 |
| 高端个人用户 | 实时音视频数字人互动，沉浸式多模态知识检索 |

---

## 2. 需求背景与研究依据

### 2.1 行业背景与前沿研究

根据 2024-2025 年最新研究成果，本项目立项依据如下：

**原生多模态实时交互已成核心范式**：
- **GPT-4o（OpenAI, 2024）**：首个原生多模态实时系统，证明端到端混合模态 Token 化相较级联 ASR→LLM→TTS 方案的绝对优势（延迟极低，情感保真）。
- **Moshi（Kyutai, 2024, arXiv:2410.00037）**：第一个全双工实时语音大模型，理论延迟 160ms，提出 Inner Monologue 机制（Decoder 在前处理流式音频）。I-Transformer 的 Decoder-first 设计直接受其启发。

**Any-to-Any 离散 Token 统一范式**：
- **AnyGPT（arXiv:2402.12226）**：通过数据层统一（不改变架构）将语音/文本/图像/音乐统一为离散 Token，单一 LLM 骨架处理所有模态。
- **Chameleon（Meta, arXiv:2405.09818）**：早期融合（Early-Fusion）混合模态模型，图文 Token 从 Embedding 层起统一处理，为本项目的统一 Token 空间设计提供直接参考。

**DiT 控制机制的可推广性**：
- **DiT（arXiv:2212.09748, ICCV 2023）**：以 Transformer 替换扩散模型 U-Net，通过 adaLN-Zero 实现强力可控生成。C-Transformer 将此机制推广到对话控制领域，实现推理中途对情绪/风格/内容的硬性干预。

**RAG 幻觉量化理论**：
- **C-RAG（arXiv:2402.03181, ICML 2024）**：首个对 RAG 模型幻觉风险进行理论认证的框架，证明当检索模型与 Transformer 质量均非平凡时，RAG 的生成风险可被证明低于单独 LLM，为无幻觉阻断机制提供理论依据。

**2025 年最新骨干模型**：
- **Qwen3（2025 年 4 月，阿里通义）**：当前开源最强 LLM 系列，QK-Norm 解决多模态混合训练梯度失配，rope_theta=1M 支持 128K 上下文，Thinking/Non-Thinking 双模式动态切换，MoE 版本（30B-A3B）单卡 A100 80G 可部署完整三分支系统。

### 2.2 用户核心痛点

1. **交互延迟高**：传统级联系统（ASR→LLM→TTS）总延迟 > 1s，无法实时打断。
2. **控制力薄弱**：Decoder-only 架构生成中途无法硬性改变风格/情感/内容走向。
3. **架构僵化**：融合 A 模型视觉理解与 B 模型语音生成能力的传统融合成本极高。
4. **幻觉频发**：多模态生成内容脱离实际业务数据，缺乏实时知识约束。

---

## 3. 系统架构总体设计

### 3.1 Tri-Transformer 核心拓扑

```
【用户端：连续多模态流】                  【控制核心】               【知识库端/输出端】
  (音频/视频/文本 混合流)                                             (音频/视频/文本 生成)
          │                                                                    ▲
          ▼                                                                    │
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────────┐
│  I-Transformer   │◄──────►│  C-Transformer   │◄──────►│   O-Transformer      │
│  正向 Dec → Enc  │        │  DiT 控制中枢    │        │   反向 Enc → Dec     │
│  · Causal Mask   │        │  · State Slots   │        │  · Planning Encoder  │
│  · Chunking Pool │        │  · CrossAttn     │        │  · adaLN-Zero 调制   │
│  · Bidir Encoder │        │  · adaLN-Zero    │        │  · Streaming Decoder │
└──────────────────┘        └──────────────────┘        └──────────────────────┘
          ▲                                                                    ▲
     (插拔大模型 A)                                                       (插拔大模型 B)
     Qwen3-8B/14B                                                        Qwen3-8B/14B
                                        ▲
                                        │ 异步实时 RAG 检索
                                        ▼
                            多模态向量数据库（Milvus）
                                LlamaIndex 编排层
```

### 3.2 信息流闭环

```
输入流(50ms切片) → I-Decoder(KV-Cache更新) → 积累M Token → I-Encoder聚合
→ i_enc → C-Transformer(状态更新) → scale/shift/gate → O-Encoder(规划+RAG注入)
→ o_plan → O-Decoder(自回归生成) → 多模态输出Token → 声学/视觉解码 → 用户端
      ↑                                    │
      └────────────── o_prev 反馈 ──────────┘
```

### 3.3 三分支拓扑与 DiT 文献对应关系

| 联合架构概念（tech_details/25） | Tri-Transformer 对应模块 |
|---|---|
| Forward-T（inversion/encoding） | I-Transformer：正向编码，把输入映射到 latent 域 |
| DiT / Reverse-T | C-Transformer：adaLN-Zero 调制中枢 |
| 串联拓扑 A（Forward→DiT→Reverse） | I→C→O 三阶段信息流，通过 State Slots 传递中间态 |
| adaLN-Zero 条件注入 | C-Transformer 对 I/O 各层进行调制 |
| EMA 稳定训练 | 三分支分阶段缝合方案（先训 C，再接 LoRA，最后联合微调） |

---

## 4. 核心模块功能需求

### 4.1 I-Transformer：正向实时输入编码（Dec → Enc）

#### 4.1.1 设计目标

实时接收连续多模态输入流（音频/视频/文本），以 0 延迟方式处理无限长序列，生成高维语义表征 `i_enc` 供 C-Transformer 消费。

#### 4.1.2 功能需求

**阶段一：Streaming Decoder（因果流处理）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| I-01 | 采用因果掩码（Causal Mask / Causal MHA），实现 0 延迟 Token-by-Token 实时接收 | P0 |
| I-02 | 支持动态 KV-Cache，维持流式低延迟，KV-Cache 随输入增量更新 | P0 |
| I-03 | 接口设计支持直接加载 HuggingFace 大模型（Qwen3-8B）前 N 层 Decoder 权重 | P0 |
| I-04 | 支持多模态 Token 混合输入（文本 BPE / 音频 Codec Token / 视觉 Token） | P0 |
| I-05 | 支持从 C-Transformer 接收控制注入信号（Ctrl Bias，加性偏置） | P1 |

**阶段二：Chunking & Pooling（分块池化）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| I-06 | 基于固定时间窗口（如每 100ms / M 个 Token）进行分块，将细粒度 Token 序列聚合为宏观表征 | P0 |
| I-07 | 支持语义边界自适应分块（可选）和固定窗口分块（默认） | P1 |
| I-08 | Chunking 触发阈值 M 可配置（默认 I 侧积累 1s 信息后触发 Encoder 聚合） | P1 |

**阶段三：Bidirectional Encoder（双向语义编码）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| I-09 | 双向全局自注意力，对 Chunking 后的宏块执行深度语义理解，输出 `i_enc` | P0 |
| I-10 | `i_enc` 维度与 C-Transformer State Slots 维度对齐（`d_model`） | P0 |
| I-11 | 支持 FlashAttention-3（或 FlashAttention-2）加速双向注意力计算 | P1 |

**技术规格（参考 Qwen3-8B 骨干）**：
- `d_model`: 4096
- `n_heads`: 32（GQA，KV heads: 8）
- `rope_theta`: 1,000,000（支持 128K 上下文）
- QK-Norm：每头独立 RMSNorm，防止多模态梯度失配

---

### 4.2 C-Transformer：DiT 生成控制中枢

#### 4.2.1 设计目标

维护全局对话状态槽（State Slots），通过 adaLN-Zero 机制动态调制 I 和 O 两端的各层特征，实现推理中途对情绪/风格/内容的硬性干预。C-Transformer 完全从随机初始化训练，是三分支"缝合"的核心。

#### 4.2.2 功能需求

**状态槽（State Slots）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| C-01 | 维护若干可学习的连续状态向量（`nn.Parameter`），作为查询向量和全局记忆锚点 | P0 |
| C-02 | 状态槽数量（N_slots）和维度（d_model）可配置 | P1 |

**交叉注意力（Cross-Attention）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| C-03 | Q = state_slot，KV_I = `i_enc`（来自 I-Transformer Encoder） | P0 |
| C-04 | Q = state_slot，KV_O = `o_prev`（来自 O-Transformer Planning Encoder 反馈） | P0 |
| C-05 | 形成 I → C → O → C 的闭合信息回路，类大脑工作记忆功能 | P0 |

**adaLN-Zero 调制（控制信号生成）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| C-06 | 通过 MLP 层（SiLU 激活）从 state_slot 生成 `scale, shift, gate` 张量 | P0 |
| C-07 | 调制参数尺寸匹配 I/O 两端各层的隐层维度，实现无侵入调制 | P0 |
| C-08 | 调制公式：`x = gate * LayerNorm(x) * (1 + scale) + shift`，gate 零初始化（adaLN-Zero） | P0 |
| C-09 | 支持推理时动态修改 state_slot 输入，瞬时切换输出风格/情感/语速 | P1 |

**幻觉监控（Fact-checking Arbiter）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| C-10 | 实时计算 `i_enc`（含 RAG 检索知识）与 `o_prev`（模型输出规划）的语义夹角 | P0 |
| C-11 | 当语义不一致超过阈值时，改变输出到 O 端的 `scale/shift` 参数，强制压低创造性生成概率 | P0 |
| C-12 | 支持触发"拒答"特征注入，迫使 Decoder 改变生成方向或直接回答"我不知道" | P1 |

---

### 4.3 O-Transformer：反向实时输出解码（Enc → Dec）

#### 4.3.1 设计目标

融合 RAG 知识与 C-Transformer 控制信号，先通过 Planning Encoder 进行全局内容规划与知识验证，再通过 Streaming Decoder 实时自回归输出多模态 Token 流。

#### 4.3.2 功能需求

**阶段一：Planning Encoder（全局规划与知识融合）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| O-01 | 双向注意力 Encoder，融合 C-Transformer 传来的控制信号（adaLN-Zero 调制） | P0 |
| O-02 | 通过交叉注意力（或前缀拼接）融合 RAG 检索回来的 Top-K 知识块（Knowledge Context） | P0 |
| O-03 | 输出 `o_plan`（规划表征）供后置 Decoder 使用 | P0 |
| O-04 | 输出 `o_prev`（池化后的规划状态），实时反馈给 C-Transformer 形成闭环 | P0 |
| O-05 | 在实际生成前执行内容验证（Verification），检验知识一致性，是无幻觉阻断的核心执行层 | P0 |

**阶段二：Streaming Decoder（流式自回归输出）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| O-06 | 因果交叉注意力：Q = 自回归 Target，KV = `o_plan`，实时自回归发射多模态 Token | P0 |
| O-07 | 接口设计支持直接加载 HuggingFace 大模型（Qwen3-8B）Decoder 权重作为生成骨干 | P0 |
| O-08 | 边生成边推流（Streaming Output），目标延迟 < 300ms（首 Token） | P0 |
| O-09 | 若预测出音频 Token，立即送入声学解码器（SNAC/EnCodec）发声 | P1 |
| O-10 | 若输出文本 Token，立即推送至前端界面（SSE/WebSocket） | P1 |

---

### 4.4 多模态统一 Tokenizer

#### 4.4.1 设计目标

实现文本/音频/视觉三种模态的统一离散 Token 空间（Any-to-Any Space），所有模态 Token 映射到共享 `d_model` 嵌入空间，通过模态标识符（Modality Embeddings）区分。

#### 4.4.2 Token 空间规划

| 模态 | Token 区间 | 编解码器 | 备注 |
|---|---|---|---|
| 文本 | `[0, 128,000)` | BPE（Qwen3 原生词表） | 标准子词分词 |
| 音频 | `[130,000, 134,000)` | SNAC / EnCodec | 多尺度 RVQ 展平或并行多头 |
| 视觉 | `[135,000, 145,000)` | SigLIP + VQ / VQ-GAN | 视频帧离散化 |
| 控制/特殊 | `[128,000, 130,000)` | - | `<\|audio_start\|>` 等 |

#### 4.4.3 功能需求

**文本 Tokenizer**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| T-01 | 复用 Qwen3 原生 BPE 词表（tiktoken），支持多语言 | P0 |

**音频 Tokenizer**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| T-02 | 集成 SNAC（Streaming Neural Audio Codec），将 16kHz 音频按 20ms/帧转换为层次化离散 Token | P0 |
| T-03 | 支持 EnCodec 作为备选音频 Codec（可配置切换） | P1 |
| T-04 | 音频 Token 序列展平后无越界（在 `[130,000, 134,000)` 区间内） | P0 |
| T-05 | 声学解码器（SNAC Decoder）支持流式实时解码，边解码边发声 | P0 |

**视觉 Tokenizer**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| T-06 | 集成 SigLIP 作为视频帧特征提取器（提供视觉-语义对齐的特征空间） | P1 |
| T-07 | 集成 VQ-GAN 进行视频帧离散化（视觉 Token 生成） | P1 |
| T-08 | 视觉 Token 无越界（在 `[135,000, 145,000)` 区间内） | P1 |

**统一 Tokenizer**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| T-09 | 提供 `UnifiedTokenizer.encode(modality, data) → input_ids` 统一接口 | P0 |
| T-10 | 提供 `UnifiedTokenizer.decode(token_ids, modality) → data` 反向接口 | P0 |
| T-11 | 所有模态 Token 映射到同一 `d_model` 嵌入空间，通过 Modality Embedding 向量区分 | P0 |

---

### 4.5 RAG 知识库与幻觉控制

#### 4.5.1 设计目标

构建支持多模态文档（PDF/视频/音频）的向量化知识库，在流式生成中以非阻塞方式实时注入知识，并通过 C-Transformer 幻觉阻断机制保证生成内容严格遵循事实。

#### 4.5.2 功能需求

**多模态文档摄入与向量化**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| R-01 | 支持 PDF/Word 文档解析（提取文本 + 图表） | P0 |
| R-02 | 支持 MP4 视频（提取关键帧 + 音频转写） | P2 |
| R-03 | 支持 MP3 音频文件（转写为文本 + 提取声学特征） | P2 |
| R-04 | 使用多模态 Embedding 模型（BGE-M3 / Nomic-Embed-Vision）将跨模态内容映射到统一向量空间 | P0 |
| R-05 | 使用 LlamaIndex 管理文档摄入、分块（Late Chunking）、嵌入全流程 | P0 |
| R-06 | 向量存储至 Milvus 2.x（支持 HNSW 索引，毫秒级 ANN 检索） | P0 |
| R-07 | 同时维护 ChromaDB 作为轻量级本地备选（开发/测试环境） | P1 |
| R-08 | 支持 BM25 关键词检索与向量检索的混合检索（Hybrid Search）+ 重排序 | P1 |

**异步实时检索回路**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| R-09 | 检索操作在后台线程/协程执行，不阻塞主线程音频生成 | P0 |
| R-10 | 当 I-Transformer 积累足够上下文（`i_enc` 更新）时，异步触发知识库查询 | P0 |
| R-11 | 检索结果（Top-K 知识块）存入共享内存/缓存队列，供 O-Transformer Planning Encoder 读取 | P0 |

**幻觉阻断与知识对齐**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| R-12 | C-Transformer 监控 `i_enc`（含检索知识）与 `o_prev` 的语义夹角，超阈值时触发干预 | P0 |
| R-13 | 干预机制：修改 scale/shift 压低"创造性"生成概率（降低 Softmax 温度） | P0 |
| R-14 | 支持"拒答"特征注入，让 Decoder 输出"根据知识库，我无法确认此信息" | P1 |
| R-15 | 幻觉率降低目标：在干扰注入测试中，模型受 C 约束后生成内容与知识库一致率 > 90% | P0 |

---

### 4.6 联合训练 Pipeline

#### 4.6.1 设计目标

实现从冷启动缝合层热启动到全量多模态对齐的分阶段联合分布式训练流程，支持 DeepSpeed ZeRO-3 + FlashAttention-3 + LoRA/QLoRA。

#### 4.6.2 训练阶段规划

**阶段 0：缝合层热启动（C-Transformer 冷启动）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| TR-01 | 加载大模型 A（I 端）和大模型 B（O 端）预训练权重，冻结所有权重 | P0 |
| TR-02 | 仅训练 C-Transformer 和 I/O 衔接层（扭合层），使用 L2/MSE Loss | P0 |
| TR-03 | 使用重构数据（文字翻译、声音模仿等简单任务）进行热启动 | P0 |

**阶段 1：模态与特征对齐（LoRA 接入）**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| TR-04 | 接入音视频 Token 语料，对大模型 A、B 注入 LoRA 旁路并开始微调 | P0 |
| TR-05 | 优化目标：生成流畅的音视频/文本混合序列，使用交叉熵损失 $L_{CE}$ | P0 |
| TR-06 | 支持 PEFT（HuggingFace）自动注入 LoRA，rank/alpha 可配置 | P0 |

**阶段 2：控制与 RAG 约束训练**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| TR-07 | 加入控制对齐损失 $L_{Ctrl}$（状态槽与特征的 MSE Loss） | P0 |
| TR-08 | 加入一致性损失 $L_{Consistency}$（对比学习，强制 `o_prev` 贴近 RAG Context） | P0 |
| TR-09 | 总损失：$L = L_{CE} + \lambda_{ctrl} \cdot L_{Ctrl} + \lambda_{cons} \cdot L_{Consistency}$，各权重可配置 | P0 |
| TR-10 | 给定不同指令状态槽（如"严格按参考文本回答"），约束 C-Transformer 输出合适的调制参数 | P1 |

**分布式训练基础设施**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| TR-11 | 集成 DeepSpeed ZeRO-3，支持跨多 GPU 的优化器状态/梯度/参数三级分片，显存节省 ~64x | P0 |
| TR-12 | 集成 FlashAttention-3（H100）或 FlashAttention-2（其他 GPU），加速长序列注意力 | P0 |
| TR-13 | 支持训练过程 EMA（指数移动平均）权重，稳定采样与指标 | P1 |
| TR-14 | 集成 Weights & Biases（W&B）监控训练指标（Loss、LR、Grad Norm） | P1 |
| TR-15 | 支持 checkpoint 保存与断点续训 | P0 |
| TR-16 | 提供 DeepSpeed/PyTorch DDP 训练脚本，至少支持 2 张 GPU 上跑通全阶段梯度更新 | P0 |

**自定义损失函数（eval/loss/ 模块）**

| 文件 | 损失函数 | 说明 |
|---|---|---|
| `hallucination_loss.py` | 幻觉检测损失 | 检测生成内容与知识库的语义偏差 |
| `rag_loss.py` | RAG 对齐损失 | $L_{Consistency}$，强制 `o_prev` 贴近检索上下文 |
| `control_alignment_loss.py` | 控制对齐损失 | $L_{Ctrl}$，验证 adaLN-Zero 调制效果 |
| `total_loss.py` | 总损失组合 | 可配置各项权重的总损失 $L$ |

---

### 4.7 推理与部署

#### 4.7.1 流式推理引擎

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| INF-01 | 实现带状态的实时推理闭环（区别于静态的 `model.generate()`） | P0 |
| INF-02 | 前端每 50ms 发送一个音视频切片，后端 Tokenizer 实时转为 Token Chunk 并送入 I-Decoder | P0 |
| INF-03 | 支持增量 KV-Cache 更新，I 侧 KV-Cache 随输入流增量维护 | P0 |
| INF-04 | 集成 vLLM + PagedAttention，消除 KV Cache 内存碎片，支持高并发低延迟服务 | P1 |
| INF-05 | 支持多请求并发推理（WebSocket/WebRTC 推流场景） | P1 |

**实时打断机制**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| INF-06 | 检测：I-Transformer 迅速捕捉用户打断信号（能量阈值或特定语义 Token） | P0 |
| INF-07 | 阻断：C-Transformer 判断需改变状态后，立即发送"强制停止"控制信号给 O-Transformer | P0 |
| INF-08 | 清空：O 侧生成 KV-Cache 被截断并刷新，开启新一轮聆听-回复循环 | P0 |
| INF-09 | 打断响应延迟目标：< 200ms（从用户发声到模型停止输出） | P0 |

**推理 CLI**

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| INF-10 | 提供 `inference_cli.py`，支持批量推理与流式输出 | P1 |
| INF-11 | 提供 `demo.py` 快速体验入口 | P1 |
| INF-12 | 提供 `verify_model.py` 模型健康检查脚本 | P0 |

#### 4.7.2 Jetson Nano 边缘部署方案

**目标**：在 NVIDIA Jetson Nano 8GB 边缘设备上训练和部署 Tri-Transformer 模型，使用 llama.cpp GGUF 量化推理。

**硬件约束**：

| 约束 | 详情 | 影响 |
|---|---|---|
| GPU | Maxwell GM20B, 128 CUDA cores, sm_53 | 无 FlashAttention/DeepSpeed/vLLM |
| CUDA | 10.2 only | PyTorch <= 1.13.x |
| 内存 | 8GB LPDDR4 共享 CPU+GPU | 约 6GB 可用于 ML |
| 架构 | aarch64 ARM Cortex-A57 | 多数预编译库不支持 |

**训练适配**：

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| JN-01 | 自动检测 Jetson Nano 硬件环境，适配训练配置（batch=1, grad_accum=4, GaLore rank=64） | P0 |
| JN-02 | GaLoreAdamW + AMP(FP16) 内存优化，训练内存 ~1.7GB（325M 轻量配置） | P0 |
| JN-03 | GradScaler init_scale=1024（Maxwell 无 Tensor Cores，保守缩放） | P0 |
| JN-04 | 实时共享内存监控，超 85% 使用率 WARNING | P1 |
| JN-05 | 仅使用轻量配置（d_model=512, ~325M 参数），QWEN3-8B 不可行 | P0 |
| JN-06 | 提供 install_jetson_deps.sh 一键安装依赖（JetPack PyTorch + llama.cpp） | P1 |

**llama.cpp GGUF 部署**：

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| LC-01 | 将 O-Transformer Streaming Decoder 权重转换为 GGUF 格式（仅单分支，I/C 非标准结构不转换） | P0 |
| LC-02 | 支持量化选项：Q4_K_M(~1.5GB), Q5_K_M(~1.8GB), Q8_0(~2.6GB) | P0 |
| LC-03 | llama.cpp server 在 aarch64 + CUDA 10.2 上编译运行（需 6 处源码 patch） | P1 |
| LC-04 | LlamaCppService 封装 llama-cpp-python，OpenAI 兼容接口 | P1 |
| LC-05 | 推理模式切换：pytorch_direct（三分支全功能） / llamacpp_gguf（单分支轻量） | P1 |
| LC-06 | 提供 convert_to_gguf.py CLI 转换入口 | P1 |

**推荐模型配置**（Jetson Nano 可行）：

| 配置 | 参数量 | FP16 模型 | GaLore训练 | Q5_K_M部署 | 可行? |
|---|---|---|---|---|---|
| 默认轻量 (d=512, vocab=151936) | ~325M | 619MB | ~1.7GB | ~1.8GB | ✅ |
| 紧凑 (d=512, vocab=32000) | ~140M | 268MB | ~1.1GB | ~0.75GB | ✅ |

---

## 5. 前端产品功能需求

### 5.1 技术栈

React 18 + TypeScript + Vite，Ant Design 5，Zustand 状态管理，Recharts 图表，Axios HTTP，Vitest 测试。

### 5.2 对话与交互页（ChatPage）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| FE-01 | 支持三种对话模式切换：文本对话、音频对话（WebRTC）、视频对话（WebRTC） | P0 |
| FE-02 | 文本对话：支持 SSE/WebSocket 流式显示模型输出（Token-by-Token 打字机效果） | P0 |
| FE-03 | 音频对话：通过 WebRTC 实时推送麦克风音频流（50ms 切片），并实时播放模型语音回复 | P1 |
| FE-04 | 显示音频可视化波形图（AudioVisualizer 组件） | P1 |
| FE-05 | 实时打断按钮：用户可随时点击或说话打断当前模型输出 | P1 |
| FE-06 | 实时风格切换滑块：调节情感/语速/知识侧重（调用 C-Transformer 控制中枢） | P2 |
| FE-07 | 显示 RAG 引用来源面板（SourcePanel），列出回复所依据的知识库条目 | P1 |
| FE-08 | 支持多会话管理（ConversationList），历史对话记录持久化 | P0 |
| FE-09 | 支持 MessageBubble 展示 Markdown / 代码高亮 / 图表 | P0 |

### 5.3 知识库管理页（DocumentsPage）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| FE-10 | 文档上传：支持 PDF/Word/MP4/MP3，显示分块与向量化进度条 | P0 |
| FE-11 | 文档列表：展示已入库文档，支持删除、搜索 | P0 |
| FE-12 | 知识库检索测试入口：输入查询语句，验证检索效果与 Top-K 结果 | P1 |

### 5.4 模型训练页（TrainingPage）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| FE-13 | 模型插件选择器：下拉菜单输入 HuggingFace Model ID，替换 I/O 两端大模型骨干 | P1 |
| FE-14 | 训练超参配置表单（TrainingConfigForm）：learning_rate、batch_size、epochs、LoRA rank 等 | P1 |
| FE-15 | 页面可触发 LoRA 微调任务，实时显示训练进度 | P1 |

### 5.5 指标监控页（MetricsPage）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| FE-16 | 实时展示训练指标：Loss（$L_{CE}$/$L_{Ctrl}$/$L_{Cons}$）、LR、Grad Norm（Recharts 折线图） | P1 |
| FE-17 | 显示训练状态卡（TrainingStatusCard）：当前 epoch、step、预计剩余时间 | P1 |
| FE-18 | 展示推理延迟、TTFT（首 Token 时间）、吞吐量等推理指标 | P2 |

### 5.6 用户认证（LoginPage / RegisterPage）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| FE-19 | JWT 认证：用户注册、登录，Token 持久化存储 | P0 |
| FE-20 | 未认证请求自动跳转登录页，登录后恢复原请求 | P0 |

---

## 6. 后端 API 需求

### 6.1 技术栈

FastAPI (Python 3.10+)，SQLAlchemy Async，Pydantic v2，JWT (python-jose)，SQLite（开发）/ PostgreSQL（生产）。

### 6.2 API 端点规范

**认证 API（/api/v1/auth/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| POST | `/register` | 用户注册，返回 JWT Token |
| POST | `/login` | 用户登录，返回 JWT Token |

**对话 API（/api/v1/chat/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| GET | `/sessions` | 获取用户所有会话列表 |
| POST | `/sessions` | 创建新会话 |
| DELETE | `/sessions/{id}` | 删除指定会话 |
| GET | `/sessions/{id}/messages` | 获取会话消息历史 |
| POST | `/sessions/{id}/messages` | 发送消息（触发 RAG + 推理） |

**流式输出 API（/api/v1/stream/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| GET | `/sse/{session_id}` | SSE 流式文本输出 |
| WS | `/ws/{session_id}` | WebSocket 双向实时通信 |
| WS | `/webrtc/signal` | WebRTC 信令交换 |

**知识库 API（/api/v1/knowledge/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| POST | `/documents` | 上传文档（异步向量化处理） |
| GET | `/documents` | 获取文档列表 |
| DELETE | `/documents/{id}` | 删除文档及其向量 |
| POST | `/search` | 知识库检索测试 |
| GET | `/documents/{id}/status` | 获取文档向量化进度 |

**模型 API（/api/v1/model/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| POST | `/infer` | 文本推理请求 |
| GET | `/status` | 模型加载状态 |
| POST | `/load` | 动态切换/加载大模型 |
| GET | `/info` | 当前模型信息（版本/参数量） |

**训练 API（/api/v1/train/）**

| 方法 | 端点 | 描述 |
|---|---|---|
| POST | `/jobs` | 创建训练任务 |
| GET | `/jobs` | 获取训练任务列表 |
| GET | `/jobs/{id}` | 获取训练状态与指标 |
| DELETE | `/jobs/{id}` | 取消训练任务 |
| GET | `/configs` | 获取预设训练配置 |

---

## 7. 评估体系需求

### 7.1 评估管道（eval/pipeline/）

| 需求 ID | 需求描述 | 优先级 |
|---|---|---|
| EV-01 | `hallucination_evaluator.py`：评估生成内容的幻觉率（与知识库的语义偏差） | P0 |
| EV-02 | `rag_evaluator.py`：评估 RAG 检索精度（Recall@K、MRR） | P0 |
| EV-03 | `dialog_evaluator.py`：评估对话质量（连贯性、指令遵循度） | P1 |
| EV-04 | `ci_gate.py`：CI 门禁，评估指标达标后才允许合并代码 | P1 |

### 7.2 评估指标

| 指标 | 定义 | 目标值 |
|---|---|---|
| 幻觉率 | RAG 干扰测试中模型坚持错误内容的比例 | < 10%（即约束有效率 > 90%） |
| TTFT（首 Token 时延） | 从收到请求到输出第一个 Token 的时间 | < 300ms |
| 打断响应延迟 | 从用户发声到模型停止输出的时间 | < 200ms |
| 音频 Tokenizer RTF | 实时因子（编解码速度 / 音频时长） | < 1.0（优于实时） |
| 检索耗时 | 知识库 Top-K 检索耗时 | < 100ms（Milvus HNSW） |
| 流式推理吞吐 | 并发请求下的 Token 生成速率 | > 100 tokens/s（单 GPU） |

---

## 8. 非功能性需求

### 8.1 性能需求

| 指标 | 要求 |
|---|---|
| 音频流式处理延迟 | I-Transformer 处理 50ms 音频切片 < 50ms（优于实时） |
| 端到端对话延迟 | 首个音频回复 Token < 300ms |
| 打断响应 | < 200ms |
| 多并发 | 支持至少 10 路 WebSocket/WebRTC 并发连接 |

### 8.2 可靠性需求

| 需求 | 要求 |
|---|---|
| 后端服务可用性 | 生产环境 99.9% SLA |
| 模型推理异常处理 | OOM/超时等异常优雅降级，返回错误提示 |
| 向量库数据持久化 | Milvus 数据持久化，重启不丢失 |
| checkpoint 恢复 | 训练中断后可从最近 checkpoint 恢复 |

### 8.3 安全需求

| 需求 | 要求 |
|---|---|
| API 认证 | 所有业务 API 需携带有效 JWT Token |
| CORS 配置 | 生产环境严格限制 CORS 来源 |
| 密钥管理 | SECRET_KEY、数据库密码等敏感信息通过环境变量注入，禁止硬编码 |
| 用户数据隔离 | 知识库文档、对话历史按用户 ID 隔离 |

### 8.4 可扩展性需求

| 需求 | 要求 |
|---|---|
| 骨干模型热替换 | 通过 API 动态切换 I/O 端大模型，无需重启服务 |
| 模态可插拔 | Phase 1 文本 → Phase 2 音频 → Phase 3 视觉，各模态 Tokenizer 模块化可独立启用 |
| 向量库可替换 | ChromaDB（开发）↔ Milvus（生产）通过配置切换，接口统一 |

### 8.5 可测试性需求

| 需求 | 要求 |
|---|---|
| 后端测试覆盖率 | pytest，核心模块覆盖率 > 80% |
| 前端测试 | Vitest + MSW Mock，覆盖主要页面和组件 |
| 模型单元测试 | 验证 TriTransformerModel 前向传播不出错，C 梯度可反传到 I/O |
| CI 自动运行 | GitHub Actions 每次 PR 自动运行测试 + lint + typecheck |

---

## 9. 技术栈选型

### 9.1 完整技术栈

| 层级 | 技术选型 | 版本 | 备注 |
|---|---|---|---|
| **骨干模型** | Qwen3-8B / 14B / 30B-MoE | 2025.04 | I/O 端插拔骨干，Apache 2.0 |
| **深度学习框架** | PyTorch | 2.x | 三分支模型实现 |
| **权重/适配器** | HuggingFace transformers + PEFT | latest | LoRA 注入 |
| **注意力加速** | FlashAttention-3 / FA-2 | latest | 长序列必须 |
| **分布式训练** | DeepSpeed ZeRO-3 | latest | 多 GPU 显存节省 ~64x |
| **推理服务** | vLLM + PagedAttention | latest | 高并发低延迟 |
| **音频编解码** | SNAC / EnCodec | latest | 实时流式音频 Token |
| **视觉编码** | SigLIP + VQ-GAN | latest | Phase 3 视觉模态 |
| **后端框架** | FastAPI | 0.100+ | Python 3.10+ |
| **ORM** | SQLAlchemy Async | 2.x | 异步数据库操作 |
| **认证** | python-jose (JWT) | latest | JWT Token 认证 |
| **向量数据库** | Milvus 2.x（生产）/ ChromaDB（开发） | latest | ANN 检索 |
| **RAG 编排** | LlamaIndex | latest | 文档摄入/检索管道 |
| **嵌入模型** | BGE-M3 / sentence-transformers | latest | 多模态嵌入 |
| **文档解析** | Unstructured / PaddleOCR | latest | PDF/图表/音频解析 |
| **前端框架** | React 18 + TypeScript + Vite | latest | 组件化 SPA |
| **UI 组件库** | Ant Design 5 | 5.x | 企业级 UI |
| **状态管理** | Zustand | latest | 轻量级状态 |
| **图表** | Recharts | latest | 训练指标可视化 |
| **实时通信** | WebSocket + WebRTC | - | 流式输出 + 音视频 |
| **前端测试** | Vitest + Testing Library + MSW | latest | 单元 + Mock 测试 |
| **后端测试** | pytest + pytest-cov + flake8 | latest | 单元 + 覆盖率 |
| **容器化** | Docker + Docker Compose | latest | 一键部署 |
| **CI/CD** | GitHub Actions | - | 自动化测试与部署 |
| **训练监控** | Weights & Biases（W&B） | latest | 训练指标记录 |

---

## 10. 阶段规划与里程碑

### Phase 1：MVP 文本模态（已完成骨架）

**目标**：跑通 Tri-Transformer 文本模态闭环，验证 I→C→O 三分支可微整体，验证左右插拔不同 LLM 的可行性。

| 里程碑 | 交付物 | 状态 |
|---|---|---|
| M1-1 | PyTorch `TriTransformerModel` 骨架（I/C/O 三分支） | 完成 |
| M1-2 | Qwen3 骨干插拔（I/O 端加载 Qwen3-8B 权重） | 完成 |
| M1-3 | 文本 BPE Tokenizer 集成 | 完成 |
| M1-4 | FastAPI 后端（认证/对话/知识库/训练 API） | 完成 |
| M1-5 | React 前端（对话/文档/训练/指标四页面） | 完成 |
| M1-6 | ChromaDB + BM25 文本 RAG 管道 | 完成 |
| M1-7 | 自定义损失函数 + 评估管道 | 完成 |
| M1-8 | Docker + CI/CD + 25 篇技术文档 | 完成 |

### Phase 2：Audio-Text 语音交互（进行中）

**目标**：引入实时音频 Token，实现双向纯语音交互（类 Moshi），对接文本 RAG 知识库，完成低延迟无幻觉语音助理。

| 里程碑 | 交付物 | 目标日期 |
|---|---|---|
| M2-1 | SNAC 音频 Tokenizer 集成，音频 Token 注入统一 Token 空间 | TBD |
| M2-2 | I-Transformer 流式音频输入管道（50ms 切片处理） | TBD |
| M2-3 | O-Transformer 流式音频输出（SNAC Decoder 实时解码） | TBD |
| M2-4 | WebRTC 前后端音频流联通（浏览器麦克风 → 模型 → 扬声器） | TBD |
| M2-5 | 实时打断机制（< 200ms 响应） | TBD |
| M2-6 | 音频模态 LoRA 对齐训练（阶段 1） | TBD |

### Phase 3：Omni-Modal 全模态数字人（规划中）

**目标**：全面接入视觉/视频流输入输出，完善多模态 RAG，发布企业级实时数字人与数字专家方案。

| 里程碑 | 交付物 | 目标日期 |
|---|---|---|
| M3-1 | SigLIP + VQ-GAN 视觉 Tokenizer 集成 | TBD |
| M3-2 | I-Transformer 视频帧流处理 | TBD |
| M3-3 | O-Transformer 视觉 Token 输出（视频合成） | TBD |
| M3-4 | Milvus 多模态向量库（图像/音频/文本混合检索） | TBD |
| M3-5 | 企业级全模态数字人方案发布 | TBD |

---

## 11. 验收标准

### 11.1 模型架构验收

| 标准 | 方法 |
|---|---|
| 前向传播不出错 | `python verify_model.py` 通过，I/C/O 三分支 forward 正常执行 |
| 梯度传播正确 | 单元测试证明 C-Transformer 的控制梯度可成功反传到 I 和 O 分支 |
| 大模型插拔成功 | 成功加载 0.5B+ 模型 A 和 B 并完成前向推理（测试用小模型） |
| adaLN-Zero 调制生效 | 修改 state_slot 后，I/O 端输出特征分布发生可测量变化 |

### 11.2 多模态 Tokenizer 验收

| 标准 | 方法 |
|---|---|
| 无越界 | 音频/视觉/文本 Token 均在指定 ID 区间内，无越界 |
| 统一映射 | 所有模态 `input_ids` 拼接为连续序列，可直接输入 Transformer |
| 流式编码可靠 | 流式 100ms 切片输入下，I 侧 KV-Cache 稳定更新 |

### 11.3 RAG 幻觉控制验收

| 标准 | 方法 |
|---|---|
| 多模态文档摄入 | 成功摄入含文字/图表/语音的文档，完成混合向量检索 |
| 非阻塞检索 | 检索在后台线程执行，音频生成无停顿（主线程不阻塞） |
| 幻觉阻断有效 | 干扰测试（注入相反 RAG 知识），模型受约束后一致率 > 90% |

### 11.4 前端功能验收

| 标准 | 方法 |
|---|---|
| 文本对话流式响应 | WebSocket 流式接收，Token-by-Token 展示，无卡顿 |
| 文档上传与检索 | 上传 PDF 后成功检索相关内容 |
| 训练任务触发 | 前端点击后触发 LoRA 微调，实时显示 Loss 曲线 |
| 认证系统 | 注册/登录/Token 续期全流程正常 |

### 11.5 性能验收

| 指标 | 目标 |
|---|---|
| 首 Token 延迟 | < 300ms（文本对话模式） |
| 打断响应 | < 200ms |
| 音频 RTF | < 1.0（SNAC 编解码优于实时） |
| CI 通过 | 所有 PR 自动通过 pytest + Vitest + lint + typecheck |

### 11.6 分布式训练验收

| 标准 | 方法 |
|---|---|
| 多 GPU 启动 | DeepSpeed 脚本在 ≥2 张 GPU 上成功启动并完成四阶段前反向梯度更新 |
| 显存控制 | ZeRO-3 分片后单卡显存占用符合预期（与理论 ~64x 节省吻合） |
| checkpoint 恢复 | 中断后从最近 checkpoint 恢复训练，Loss 曲线连续 |
