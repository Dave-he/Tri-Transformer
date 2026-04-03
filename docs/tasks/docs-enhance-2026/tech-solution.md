# 技术方案 — docs-enhance-2026

## 方案概述

**策略**：追加新节，不修改现有内容
**变更文件数**：8 篇 Markdown 文档

## 变更清单

### P0 文档（各新增 ≥50 行）

| 文件 | 新增小节 | 技术重点 |
|------|---------|---------|
| `09_vqgan.md` | §5.5-5.7 | COSMOS Tokenizer（NVIDIA）、OpenMAGVIT2、LlamaGen |
| `10_siglip.md` | §5.4-5.6 | SigLIP 2、InternVL3、Qwen2-VL 推理代码 |
| `12_anygpt_any2any.md` | §5.x | Emu3、Janus-Pro、Show-o |
| `15_milvus.md` | §5.x-6.x | Milvus 2.5 稀疏向量、GPU 索引、混合检索 |
| `16_llamaindex.md` | §5.x | Workflows、Multi-agent、LlamaParse |
| `21_webrtc.md` | §6.x-7.x | WHIP/WHEP、aiortc 完整实践 |

### P1 文档（各新增 ≥30 行）

| 文件 | 新增内容 |
|------|---------|
| `03_bidirectional_encoder.md` | ModernBERT 全局+局部注意力细节、GTE-Qwen2-7B 实践 |
| `02_chunking_pooling.md` | MegaByte/MEGALODON、超长上下文最新进展 |

## 实施顺序

P0 优先，从最短文档开始（降低上下文占用风险）：
1. 09_vqgan.md（192行）
2. 10_siglip.md（198行）
3. 12_anygpt_any2any.md（195行）
4. 15_milvus.md（229行）
5. 16_llamaindex.md（229行）
6. 21_webrtc.md（279行）
7. 03_bidirectional_encoder.md（P1）
8. 02_chunking_pooling.md（P1）
