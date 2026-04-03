# Tri-Transformer 文档中心

欢迎来到 Tri-Transformer 项目的文档中心！本文档索引将帮助您快速找到所需信息。

---

## 📚 文档分类

### 🚀 入门指南

| 文档 | 描述 | 阅读时间 |
|------|------|---------|
| [README.md](../README.md) | 项目概览、特性介绍、快速导航 | 5 分钟 |
| [QUICKSTART.md](./QUICKSTART.md) | 5 分钟快速上手指南 | 5 分钟 |
| [INSTALLATION.md](./installation/INSTALLATION.md) | 详细的安装部署指南 | 15 分钟 |
| [FAQ.md](./FAQ.md) | 常见问题解答 | 按需查阅 |

### 📖 核心文档

| 文档 | 描述 | 阅读时间 |
|------|------|---------|
| [产品需求文档](./Tri-Transformer 可控对话与 RAG 知识库增强系统.md) | 完整的产品需求说明 | 30 分钟 |
| [技术调研报告](./技术调研报告.md) | 核心技术名词系统性综述 | 60 分钟 |
| [API 参考](./API_REFERENCE.md) | 完整的 API 接口文档 | 按需查阅 |
| [CHANGELOG.md](../CHANGELOG.md) | 版本变更日志 | 按需查阅 |

### 🏗️ 开发文档

| 文档 | 描述 | 阅读时间 |
|------|------|---------|
| [系统架构](./agent/architecture.md) | 模块结构与架构设计 | 15 分钟 |
| [开发命令](./agent/development_commands.md) | 构建、测试、运行命令参考 | 5 分钟 |
| [测试规范](./agent/testing.md) | 测试框架与约定 | 10 分钟 |
| [开发规范](./agent/conventions.md) | 代码风格与目录约定 | 10 分钟 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 | 15 分钟 |

### 🔬 技术深潜（23 篇）

#### I-Transformer 模块

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [01_causal_mask.md](./tech_details/01_causal_mask.md) | Causal Mask（因果掩码） | 10 分钟 |
| [02_chunking_pooling.md](./tech_details/02_chunking_pooling.md) | Chunking & Pooling | 10 分钟 |
| [03_bidirectional_encoder.md](./tech_details/03_bidirectional_encoder.md) | Bidirectional Encoder | 10 分钟 |

#### C-Transformer 模块

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [04_dit.md](./tech_details/04_dit.md) | DiT（Diffusion Transformer） | 15 分钟 |
| [05_adaln_zero.md](./tech_details/05_adaln_zero.md) | adaLN-Zero | 10 分钟 |
| [06_cross_attention_state_slots.md](./tech_details/06_cross_attention_state_slots.md) | Cross-Attention & State Slots | 15 分钟 |

#### 多模态 Token 化

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [07_encodec.md](./tech_details/07_encodec.md) | EnCodec | 10 分钟 |
| [08_snac.md](./tech_details/08_snac.md) | SNAC | 10 分钟 |
| [09_vqgan.md](./tech_details/09_vqgan.md) | VQ-GAN | 10 分钟 |
| [10_siglip.md](./tech_details/10_siglip.md) | SigLIP | 10 分钟 |
| [11_bpe.md](./tech_details/11_bpe.md) | BPE | 10 分钟 |
| [12_anygpt_any2any.md](./tech_details/12_anygpt_any2any.md) | AnyGPT & Any-to-Any | 15 分钟 |
| [13_chameleon.md](./tech_details/13_chameleon.md) | Chameleon | 15 分钟 |

#### RAG 知识库

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [14_rag.md](./tech_details/14_rag.md) | RAG 体系 | 15 分钟 |
| [15_milvus.md](./tech_details/15_milvus.md) | Milvus | 10 分钟 |
| [16_llamaindex.md](./tech_details/16_llamaindex.md) | LlamaIndex | 10 分钟 |

#### 训练与部署

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [17_deepspeed_zero3.md](./tech_details/17_deepspeed_zero3.md) | DeepSpeed ZeRO-3 | 15 分钟 |
| [18_flashattention3.md](./tech_details/18_flashattention3.md) | FlashAttention-3 | 10 分钟 |
| [19_lora_qlora.md](./tech_details/19_lora_qlora.md) | LoRA & QLoRA | 15 分钟 |
| [20_vllm_pagedattention.md](./tech_details/20_vllm_pagedattention.md) | vLLM & PagedAttention | 10 分钟 |
| [21_webrtc.md](./tech_details/21_webrtc.md) | WebRTC | 10 分钟 |

#### 前沿架构对比

| 文档 | 技术名词 | 阅读时间 |
|------|---------|---------|
| [22_frontier_models.md](./tech_details/22_frontier_models.md) | GPT-4o / Moshi / Qwen2-Audio 等 | 20 分钟 |
| [23_qwen3.md](./tech_details/23_qwen3.md) | Qwen3 骨干模型 | 15 分钟 |

### 📋 子需求文档

| 文档 | 描述 | 阅读时间 |
|------|------|---------|
| [01 模型骨架与插拔 LLM](./sub_prds/01_model_skeleton_and_pluggable_llm.md) | Tri-Transformer 模型骨架设计 | 20 分钟 |
| [02 多模态 Token 化与流式引擎](./sub_prds/02_multimodal_tokenizer_and_streaming_engine.md) | 音频/视频 Token 化方案 | 20 分钟 |
| [03 实时多模态 RAG 与幻觉控制](./sub_prds/03_realtime_multimodal_rag_and_hallucination_control.md) | RAG 与幻觉检测机制 | 20 分钟 |
| [04 训练管线与全双工前端](./sub_prds/04_training_pipeline_and_full_duplex_frontend.md) | 训练系统与前端实现 | 20 分钟 |

### 🔧 研发资产

| 文档 | 描述 |
|------|------|
| [rd-assets.md](./research/rd-assets.md) | 研发资产索引报告 |
| [project-conventions.yaml](./research/project-conventions.yaml) | 项目约定 |
| [api-contracts.yaml](./research/api-contracts.yaml) | API 契约 |
| [domain-model.yaml](./research/domain-model.yaml) | 领域模型 |

---

## 🎯 快速导航

### 按使用场景

#### 我是新手，想快速体验
1. [README.md](../README.md) - 了解项目
2. [QUICKSTART.md](./QUICKSTART.md) - 5 分钟上手
3. [FAQ.md](./FAQ.md) - 遇到问题先查这里

#### 我要部署项目
1. [INSTALLATION.md](./installation/INSTALLATION.md) - 安装指南
2. [API_REFERENCE.md](./API_REFERENCE.md) - API 文档
3. [FAQ.md](./FAQ.md) - 故障排查

#### 我要贡献代码
1. [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
2. [开发规范](./agent/conventions.md) - 代码规范
3. [测试规范](./agent/testing.md) - 测试要求

#### 我要深入了解技术
1. [技术调研报告](./技术调研报告.md) - 技术综述
2. [系统架构](./agent/architecture.md) - 架构设计
3. [技术深潜文档](./tech_details/README.md) - 23 篇技术详解

#### 我要开发新功能
1. [产品需求文档](./Tri-Transformer 可控对话与 RAG 知识库增强系统.md) - 需求说明
2. [子需求文档](./sub_prds/) - 详细设计
3. [API 参考](./API_REFERENCE.md) - 接口定义

---

## 📊 文档统计

| 类别 | 文档数量 | 总阅读时间 |
|------|---------|-----------|
| 入门指南 | 4 篇 | ~25 分钟 |
| 核心文档 | 4 篇 | ~90 分钟 |
| 开发文档 | 5 篇 | ~55 分钟 |
| 技术深潜 | 23 篇 | ~300 分钟 |
| 子需求 | 4 篇 | ~80 分钟 |
| **总计** | **40+ 篇** | **~9 小时** |

---

## 🔍 搜索技巧

### 在 GitHub 上搜索

```
在仓库中搜索：site:github.com/your-org/tri-transformer 关键词
```

### 使用 GitHub 搜索

1. 访问仓库主页
2. 按 `t` 键打开文件搜索
3. 输入文件名或关键词

### 本地搜索

```bash
# 搜索文档内容
grep -r "关键词" docs/

# 使用 ripgrep（更快）
rg "关键词" docs/
```

---

## 📝 文档更新记录

### 2026-04-03
- ✅ 创建 README.md 主文档
- ✅ 创建 INSTALLATION.md 安装指南
- ✅ 创建 QUICKSTART.md 快速开始
- ✅ 创建 API_REFERENCE.md API 参考
- ✅ 创建 FAQ.md 常见问题
- ✅ 创建 CONTRIBUTING.md 贡献指南
- ✅ 创建 CHANGELOG.md 变更日志

### 待更新
- 📋 性能基准报告
- 📋 模型架构可视化图表
- 📋 视频教程链接
- 📋 最佳实践案例

---

## 💡 文档贡献

欢迎帮助改进文档！

### 可以贡献的内容

- ✏️ 修复拼写错误
- 📝 补充缺失说明
- 🌍 改进翻译质量
- 💡 添加示例代码
- 🎨 改进排版格式

### 如何贡献

1. Fork 项目仓库
2. 创建文档分支：`git checkout -b docs/your-improvement`
3. 修改文档
4. 提交更改：`git commit -m "docs: improve xxx documentation"`
5. 推送并创建 Pull Request

详见：[CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 🔗 外部资源

### 相关项目文档

- [Qwen3 文档](https://qwen.readthedocs.io/) - 骨干模型文档
- [FastAPI 文档](https://fastapi.tiangolo.com/) - 后端框架
- [React 文档](https://react.dev/) - 前端框架
- [PyTorch 文档](https://pytorch.org/docs/) - 深度学习框架

### 技术社区

- [Hugging Face](https://huggingface.co/) - 模型与数据集
- [Papers With Code](https://paperswithcode.com/) - 最新论文与代码
- [知乎 - AI 话题](https://www.zhihu.com/topic/17779887) - 中文 AI 社区

---

## 📞 获取帮助

如果文档未能解决您的问题：

1. 💬 查看 [FAQ.md](./FAQ.md)
2. 🔍 搜索 [GitHub Issues](https://github.com/your-org/tri-transformer/issues)
3. 📝 创建新 Issue
4. 📧 联系：support@example.com

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team

---

<div align="center">

**Tri-Transformer 文档中心**

[返回顶部](#tri-transformer 文档中心)

</div>
