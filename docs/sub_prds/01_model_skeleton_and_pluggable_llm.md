# 子需求文档 1：Tri-Transformer 模型骨架与双端大模型插拔系统

## 1. 概述
本需求详细说明了 Tri-Transformer 核心架构（I-C-O 三分支）的底层代码骨架实现，以及如何支持在训练和推理阶段，左右两端（I 和 O 分支）插拔不同的开源大模型权重（大模型 A 与 大模型 B）。这是整个项目最基础的底层工程。

## 2. 核心需求目标
1. **模型骨架搭建**：实现 `I-Transformer` (Dec->Enc)、`C-Transformer` (DiT-style)、`O-Transformer` (Enc->Dec) 的 PyTorch/HuggingFace `nn.Module` 代码骨架。
2. **大模型权重兼容层**：设计 Adapter 接口，允许加载 HuggingFace 上的标准模型（如 Qwen2、Llama3）的特定层作为 I 和 O 分支的 Decoder/Encoder。
3. **扭合层设计（Coupling Layers）**：实现 I、C、O 之间的交叉注意力（Cross-Attention）和控制信号调制（adaLN-Zero）模块。

## 3. 功能详细设计

### 3.1 I-Transformer 骨架设计
- **功能**：处理实时连续的多模态 Token 序列流。
- **阶段 1：Streaming Decoder**
  - **接口**：接入大模型 A（如 Qwen2-Audio/VL）。
  - **实现**：截取大模型 A 的前 $N$ 层因果 Transformer Decoder 层。支持动态 KV-Cache 以维持流式低延迟输入。
- **阶段 2：Chunking & Encoder**
  - **实现**：滑动窗口池化（Pooling）将 Token 流压缩。后接双向 Transformer Encoder 层，生成全局上下文表征 `i_enc`。

### 3.2 C-Transformer 骨架设计
- **功能**：全局状态控制中枢。
- **实现**：纯随机初始化的自研模块。
  - **状态槽 (State Slots)**：可学习的 `nn.Parameter`，作为查询向量（Query）。
  - **交叉注意力**：接收 `i_enc` (来自 I) 和 `o_prev` (来自 O) 作为 Key-Value。
  - **adaLN-Zero 发生器**：通过 MLP 层输出 `scale`, `shift`, `gate` 张量，尺寸匹配 I 和 O 的隐层维度。

### 3.3 O-Transformer 骨架设计
- **功能**：结合控制信号与知识，自回归输出多模态 Token。
- **阶段 1：Planning Encoder**
  - **实现**：双向 Encoder 层。融合 RAG 检索回来的文本/跨模态 Context 嵌入。
  - **扭合机制**：接收 C-Transformer 传来的 adaLN-Zero `scale/shift` 调制。
  - **反馈机制**：该层的输出池化后作为 `o_prev` 传回 C-Transformer。
- **阶段 2：Streaming Decoder**
  - **接口**：接入大模型 B（如 Llama3、VALL-E 等生成模型）。
  - **实现**：截取大模型 B 的自回归 Decoder 层。接收 Encoder 层的输出作为 Cross-Attention 的 Key-Value。生成最终的概率分布（Logits）。

### 3.4 双端大模型插拔系统
- **动态加载器**：根据配置字典（JSON/YAML），自动从本地或 HF Hub 下载并映射大模型 A 和 B 的权重到 I 和 O 的相应层。
- **权重冻结与 LoRA**：支持对插拔的大模型权重进行冻结，并自动注入 LoRA 旁路适配器进行轻量微调；保证 C-Transformer 和扭合层全量训练。

## 4. 技术栈建议
- 框架：PyTorch 2.x
- 底层生态：HuggingFace `transformers` (用于权重加载和 Tokenizer)、`peft` (用于 LoRA)
- 算子优化：FlashAttention-2/3 (支持 Causal 与非 Causal 模式)

## 5. 验收标准
1. 提供 `TriTransformerModel` 的前向传播伪代码和真实 PyTorch 类。
2. 单元测试证明：能成功加载一个 0.5B 级别的模型 A 和另一个 0.5B 级别的模型 B 并完成前向推理不出错。
3. 单元测试证明：C-Transformer 的控制梯度可以成功反向传播到 I 和 O 分支。
