# 子需求文档 2：跨模态统一 Tokenizer 与流式前向推理引擎

## 1. 概述
为实现多模态（文本、音频、视频）的统一处理以及“边听边想边说”的全双工实时能力，必须设计统一的 Any-to-Any 离散 Token 空间，并开发支持流式（Streaming）状态保持的前向推理引擎。

## 2. 核心需求目标
1. **多模态 Tokenizer**：集成文本 BPE、音频 Neural Codec、视觉 VQ-GAN，映射到统一的离散 ID 空间。
2. **流式输入/输出管道（Pipeline）**：支持从麦克风/摄像头获取数据流转 Token，以及从 Logits 转声音/视频画面并输出。
3. **状态化前向引擎**：实现支持 KV-Cache 滑动窗口的实时前向推理逻辑。

## 3. 功能详细设计

### 3.1 跨模态统一 Token 空间 (Any-to-Any Space)
- **文本模态**：采用大模型 B 的原生文本词表（如 128,000）。
- **音频模态**：集成 SNAC 或 EnCodec。将音频波形（例如 16kHz）按照 20ms 一帧转换为几层离散 Token，做展平处理或并行多头生成。音频 Token 占用保留区间 `[130000, 134000]`。
- **视觉模态**：集成 SigLIP + VQ。将视频帧转换为视觉 Token，占用保留区间 `[135000, 145000]`。
- **控制/特殊 Token**：定义特殊分隔符，如 `<|audio_start|>`, `<|vision_start|>`, `<|interrupt|>` 等。

### 3.2 增量输入与流式前向推理引擎 (Streaming Engine)
普通的 `model.generate()` 是静态的。需要实现带状态的实时推理闭环：
- **实时接收循环**：
  - 前端通过 WebRTC 发送每 50ms 的原始音视频流，后端 Tokenizer 实时转成 Token Chunk (如大小为 N)。
  - 将 Token Chunk 送入 `I-Transformer.Decoder`，更新 I 侧 KV-Cache。
- **中枢规划触发器**：
  - 每当 I 侧积累了 $M$ 个 Token（如 1 秒钟的信息），触发一次 I-Encoder 聚合，生成最新的 `i_enc`。
  - 将 `i_enc` 送入 `C-Transformer` 计算最新的控制信号和 `scale/shift`。
- **实时输出自回归**：
  - `O-Transformer.Decoder` 在后台不断以自回归形式预测下一个 Token。
  - 若预测出音频 Token，立即送入声学解码器发声；若输出文本 Token，则推送到前端界面。

### 3.3 实时打断机制 (Interruption Handling)
- **检测**：当用户在模型说话时突然发声，I-Transformer 迅速捕捉到能量或特定语义（如“等等”）。
- **阻断**：C-Transformer 判断需要改变状态，立即发送“强制停止”控制信号给 O-Transformer。
- **清空**：O 侧的生成 KV-Cache 被截断并刷新，开启新一轮的聆听-回复循环。

## 4. 技术栈建议
- 音频 Tokenizer：`snac` (Streaming Neural Audio Codec) 或 `encodec`
- 视觉 Tokenizer：`Llama-3-Vision` 的图像 Encoder 部分的离散化组件
- 推理引擎底座：基于 `vLLM` 改造或手写 PyTorch 异步生成队列。

## 5. 验收标准
1. 音视频/文本数据能被统一映射到同一个连续的 `input_ids` 序列且不越界。
2. 流式前向引擎能在模拟 100ms 切片的音频输入下，在后台稳定更新 I 侧 KV-Cache。
3. 实现打断测试：当注入中断 Token 时，O 侧生成能够立即（延迟 < 200ms）终止并切换。
