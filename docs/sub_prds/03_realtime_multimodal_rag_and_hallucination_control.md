# 子需求文档 3：实时响应的跨模态 RAG 知识库与事实校验中枢

## 1. 概述
在全双工多模态生成中，传统检索带来的高延迟和文不对题是致命的。本项目通过 O-Transformer 的前置 Encoder（Planning Encoder）以及控制中枢，实现多模态知识片段（图像/音频/文本）的实时注入和无幻觉事实约束。

## 2. 核心需求目标
1. **多模态向量库构建**：支持多模态文档（含图片、语音纪要）的分块和统一向量化存储。
2. **零延迟检索注入机制**：在流式生成中，以非阻塞方式进行知识库的连续异步查询。
3. **幻觉阻断与知识对齐**：当模型生成的音频/文本偏离知识库事实时，通过 C-Transformer 动态干预生成概率分布。

## 3. 功能详细设计

### 3.1 多模态摄入与向量化 (Multimodal Ingestion)
- **支持格式**：PDF/Word (提取文本+图像)、MP4 (提取关键帧+音频转写)、MP3。
- **Late Chunking & 统一 Embedding**：
  - 对提取的多模态数据，通过类似 CLIP 或 BGE-M3 的多模态 Embedding 模型，将跨模态内容映射到统一的向量空间（如 768 或 1024 维）。
  - 存储入多模态向量数据库（如 Milvus）。

### 3.2 异步实时检索回路 (Async Retrieval Loop)
由于音视频实时流不能阻塞等待检索结果：
- **前置预检**：在 I-Transformer 积累一定上下文 `i_enc` 时，系统触发异步检索引擎。
- **Context 缓存**：检索回来的 Top-K 个跨模态 Knowledge Context (包含 Token IDs 和 密集表征 Dense Vectors) 被放入共享内存或缓存队列。

### 3.3 RAG 与 O-Transformer 结合 (Knowledge Planning)
- **规划融合**：当 O-Transformer 的前置 Encoder (Planning Encoder) 运行时，它会读取当前最新的 Knowledge Context。
- **Attention 交互**：Encoder 通过交叉注意力（或直接拼接前缀）关注这些知识特征，计算出受知识锚定的 `o_plan`，并将此表征提供给后置 Decoder 进行具体发音/写字。

### 3.4 事实校验与动态干预 (Fact-checking & Intervention)
- **不一致性检测**：C-Transformer 监控 `i_enc` (输入+检索知识) 和 `o_prev` (模型准备输出的规划)。如果两者在语义空间夹角过大（即存在编造倾向）。
- **控制信号抑制**：C-Transformer 改变输出到 O 端的 `scale/shift` 参数，强行压低“创造性”生成的概率（如降低 Softmax 温度，或直接注入拒答特征），迫使 Decoder 改变生成方向或直接回答“我不知道”。

## 4. 技术栈建议
- 多模态嵌入：BGE-M3, Nomic-Embed-Vision
- 向量数据库：Milvus 2.x
- 文档解析：Unstructured, PaddleOCR
- 检索引擎：LlamaIndex (定制异步接口)

## 5. 验收标准
1. 能成功摄入包含文字、图表和语音片段的文档，并完成多模态混合向量检索。
2. 检索耗时应脱离主线程（音频生成不因检索停顿），通过后台线程注入。
3. 注入干扰测试：故意提供相反的 RAG 知识，模型必须受 C-Transformer 约束放弃原有倾向，转而根据新知识生成回复（幻觉率降低 > 90%）。
