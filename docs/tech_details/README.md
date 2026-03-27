# Tri-Transformer 子技术文档索引

本目录为《Tri-Transformer 可控对话与 RAG 知识库增强系统》技术调研报告的深入展开，共 22 篇子技术文档，覆盖 PRD 与调研报告中全部核心技术名词。

---

## 模块一：I-Transformer — 正向实时输入编码

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [01_causal_mask.md](./01_causal_mask.md) | Causal Mask（因果掩码） | 上三角掩码原理、KV Cache 流式推理、GQA/RoPE 最新进展 |
| [02_chunking_pooling.md](./02_chunking_pooling.md) | Chunking & Pooling（分块池化） | 固定窗口/语义边界分块、注意力池化、StreamingLLM |
| [03_bidirectional_encoder.md](./03_bidirectional_encoder.md) | Bidirectional Encoder（双向编码器） | BERT 架构、全局自注意力、ModernBERT 最新进展 |

---

## 模块二：C-Transformer — DiT 生成控制中枢

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [04_dit.md](./04_dit.md) | DiT（Diffusion Transformer） | Transformer 替换 U-Net、规模律、SD3/FLUX/CogVideoX |
| [05_adaln_zero.md](./05_adaln_zero.md) | adaLN-Zero（自适应层归一化） | 零初始化机制、实时可控性、Diffusion Forcing 最新进展 |
| [06_cross_attention_state_slots.md](./06_cross_attention_state_slots.md) | Cross-Attention & State Slots | Perceiver IO 灵感、闭环信息流设计、Jamba/Flamingo 最新进展 |

---

## 模块三：多模态 Token 化与对齐

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [07_encodec.md](./07_encodec.md) | EnCodec（神经音频编解码器） | RVQ 残差量化、流式架构、DAC/SNAC/SemanticCodec 进展 |
| [08_snac.md](./08_snac.md) | SNAC（多尺度神经音频编解码器） | 多尺度量化、层次 Token 树、Orpheus TTS/Hertz-Dev 应用 |
| [09_vqgan.md](./09_vqgan.md) | VQ-GAN（向量量化生成对抗网络） | CNN+VQ+Transformer、视频 Token 化、COSMOS Tokenizer |
| [10_siglip.md](./10_siglip.md) | SigLIP（Sigmoid 语言图像预训练） | Sigmoid vs Softmax 损失、PaliGemma/InternVL/Qwen2-VL 应用 |
| [11_bpe.md](./11_bpe.md) | BPE（字节对编码） | 子词合并算法、tiktoken、多模态 Token 空间扩展 |
| [12_anygpt_any2any.md](./12_anygpt_any2any.md) | AnyGPT & Any-to-Any 范式 | 数据层统一多模态、模态速率失配、Unified-IO 2/Emu3 |
| [13_chameleon.md](./13_chameleon.md) | Chameleon（早期融合混合模态） | 早期融合 vs 晚期融合、QK-Norm 训练稳定化、Emu3/LWM |

---

## 模块四：RAG 知识库增强

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [14_rag.md](./14_rag.md) | RAG（检索增强生成）体系 | Naive/Advanced/Modular 三范式、无幻觉阻断、GraphRAG |
| [15_milvus.md](./15_milvus.md) | Milvus（多模态向量数据库） | 云原生架构、HNSW/IVF/DiskANN 索引、Milvus 2.4 稀疏向量 |
| [16_llamaindex.md](./16_llamaindex.md) | LlamaIndex（RAG 编排框架） | 完整 RAG Pipeline、Milvus 集成、LlamaAgents/Workflows |

---

## 模块五：训练与部署技术栈

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [17_deepspeed_zero3.md](./17_deepspeed_zero3.md) | DeepSpeed ZeRO-3 | 三阶段分片、CPU/NVMe Offload、显存估算、FSDP 对比 |
| [18_flashattention3.md](./18_flashattention3.md) | FlashAttention-3 | Warp 专化异步流水线、FP8 量化、740 TFLOPs/s、Ring Attention |
| [19_lora_qlora.md](./19_lora_qlora.md) | LoRA & QLoRA（低秩适配微调） | 低秩分解原理、NF4 量化、Tri-Transformer 分阶段缝合训练方案 |
| [20_vllm_pagedattention.md](./20_vllm_pagedattention.md) | vLLM & PagedAttention | KV Cache 分页、连续批处理、流式 WebSocket 推理、v0.6.0 性能 |
| [21_webrtc.md](./21_webrtc.md) | WebRTC（全双工实时通信） | ICE/STUN/TURN、Opus 音频、aiortc 服务端实现、WHIP/WHEP |

---

## 模块六：前沿架构对比

| 文档 | 技术名词 | 核心内容 |
|---|---|---|
| [22_frontier_models.md](./22_frontier_models.md) | GPT-4o / Moshi / VALL-E 2 / Qwen2-Audio / AnyGPT / Chameleon | 六大系统架构对比矩阵、Tri-Transformer 差异化分析 |

---

## 快速导航

**按技术类别**：
- 注意力机制：[01](./01_causal_mask.md)、[03](./03_bidirectional_encoder.md)、[06](./06_cross_attention_state_slots.md)、[18](./18_flashattention3.md)
- 音频 Token 化：[07](./07_encodec.md)、[08](./08_snac.md)
- 视觉 Token 化：[09](./09_vqgan.md)、[10](./10_siglip.md)
- 多模态统一：[11](./11_bpe.md)、[12](./12_anygpt_any2any.md)、[13](./13_chameleon.md)
- 可控生成：[04](./04_dit.md)、[05](./05_adaln_zero.md)
- 知识库 RAG：[14](./14_rag.md)、[15](./15_milvus.md)、[16](./16_llamaindex.md)
- 高效训练：[17](./17_deepspeed_zero3.md)、[19](./19_lora_qlora.md)
- 高效推理：[20](./20_vllm_pagedattention.md)
- 实时通信：[21](./21_webrtc.md)

**按开发阶段**：
- Phase 1（MVP 文本模态）：[01](./01_causal_mask.md)、[02](./02_chunking_pooling.md)、[03](./03_bidirectional_encoder.md)、[04](./04_dit.md)、[05](./05_adaln_zero.md)、[11](./11_bpe.md)、[17](./17_deepspeed_zero3.md)、[19](./19_lora_qlora.md)
- Phase 2（音频对话）：[07](./07_encodec.md)、[08](./08_snac.md)、[14](./14_rag.md)、[15](./15_milvus.md)、[16](./16_llamaindex.md)、[18](./18_flashattention3.md)、[20](./20_vllm_pagedattention.md)
- Phase 3（全模态数字人）：[09](./09_vqgan.md)、[10](./10_siglip.md)、[12](./12_anygpt_any2any.md)、[13](./13_chameleon.md)、[21](./21_webrtc.md)
