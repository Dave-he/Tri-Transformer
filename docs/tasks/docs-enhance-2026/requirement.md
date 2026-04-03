# 需求文档 — docs-enhance-2026

## 任务概述

**任务 ID**: docs-enhance-2026
**类型**: documentation
**平台**: Backend（Python/PyTorch）
**质量分**: 88

## 背景

Tri-Transformer 项目的技术文档体系（`docs/tech_details/`）已覆盖 25 篇核心技术文档，
但部分文档的"最新进展"章节尚不完整，特别是 2024-2025 年间涌现的重要开源模型和工程实践。
本任务通过深度补充文档内容，提升技术参考价值。

## 增强目标

### P0 目标文档（primary_targets）

| 文档 | 增强重点 |
|------|---------|
| `09_vqgan.md` | COSMOS Tokenizer（NVIDIA 2024）、OpenMAGVIT2、LlamaGen 实践细节 |
| `10_siglip.md` | SigLIP 2（2024）、InternVL3（2025）、Qwen2-VL 集成完整代码示例 |
| `12_anygpt_any2any.md` | Emu3（2024）、Janus-Pro（2025）、Show-o（Wisconsin）最新进展 |
| `15_milvus.md` | Milvus 2.5 稀疏向量（BM25 融合）、GPU IVF 索引、混合检索实践 |
| `16_llamaindex.md` | LlamaIndex Workflows（0.10+）、Multi-agent框架、LlamaParse、AgentSearch |
| `21_webrtc.md` | WHIP/WHEP 协议、aiortc 完整服务端实现、SFU 架构、媒体处理流水线 |

### P1 目标文档（secondary_targets）

| 文档 | 增强重点 |
|------|---------|
| `03_bidirectional_encoder.md` | ModernBERT 深化（全局+局部注意力）、NomicBERT、GTE-Qwen2-7B |
| `02_chunking_pooling.md` | MegaByte（字节级分层）、MEGALODON（超长序列线性复杂度）实践 |

## 验收标准

1. 每篇 P0 文档新增 ≥ 50 行，含可运行代码示例和论文引用
2. 每篇 P1 文档新增 ≥ 30 行
3. 新增内容明确说明与 Tri-Transformer 的集成关联
4. 代码示例使用 PyTorch/Python 且风格与现有代码一致

## 约束

- 追加方式：在现有章节末尾新增小节，不修改已有内容结构
- 不引入无法验证的技术细节
