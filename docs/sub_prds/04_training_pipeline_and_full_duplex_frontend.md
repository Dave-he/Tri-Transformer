# 子需求文档 4：端到端联合训练 Pipeline 与平台化前端界面

## 1. 概述
为支撑 Tri-Transformer 的特殊架构，需建立一套分阶段的联合分布式训练（Joint Training）流程。同时，开发对应的管理界面（前端）以便用户上传知识库、配置大模型插件和进行实时对话交互。

## 2. 核心需求目标
1. **多阶段联合训练 Pipeline**：实现冻结/解冻逻辑，完成模型从预训练冷启动、LoRA 接入、到全量多模态对齐的训练流程。
2. **前后端交互系统**：提供 Web 端操作入口，支持配置、监控训练进度、全双工 WebRTC 音视频交互。

## 3. 功能详细设计

### 3.1 联合训练 Pipeline (Training Pipeline)
按照 PRD 中的四阶段策略编写 PyTorch DDP / DeepSpeed 脚本：
- **阶段 0：缝合层热启动**
  - 加载大模型 A 和 B，冻结所有权重。
  - 仅使用重构数据（如文字翻译或单纯的声音模仿任务），利用 L2/MSE Loss 训练中间的 C-Transformer 和 O-Encoder。
- **阶段 1：模态与特征对齐 (Alignment)**
  - 接入音视频 Token 语料。
  - 大模型 A、B 接入 LoRA 旁路。开始微调两侧的 Decoder/Encoder，目标是生成流畅的音视频/文本混合序列。
- **阶段 2：控制与 RAG 约束训练 (Control & Consistency)**
  - 加入强化控制损失函数（Control Loss）。
  - 给定不同指令状态槽（如“要求回答带愤怒情绪”、“要求严格按参考文本回答”），约束 C-Transformer 输出合适的 `scale/shift` 参数。
- **损失函数集**：
  - $L_{CE}$ (自回归交叉熵, 文本/音视频离散 Token 预测)
  - $L_{Ctrl}$ (状态与特征对齐的 MSE Loss)
  - $L_{Consistency}$ (对比学习损失，强制 $o_{prev}$ 贴近 $RAG_{Context}$)

### 3.2 训练平台后端 API
- 基于 FastAPI，提供模型加载、训练超参配置、启动进程任务的 API 接口。
- 采用 TensorBoard 或 Weights & Biases (W&B) 回调监听训练 metrics（Loss, LR, Grad Norm）。

### 3.3 全双工用户交互前端界面 (Frontend Dashboard)
- **对话/交互页**：
  - 基于 WebRTC 实现麦克风、摄像头的实时推流。
  - 画布区域实时显示系统的响应（文字瀑布流、音频播放波形图或视频生成画面）。
  - 支持快捷打断按钮和实时风格切换滑块（调用控制中枢）。
- **RAG 管理页**：
  - 文档/音视频上传，显示分块与向量化进度。
  - 知识库检索测试入口（验证检出效果）。
- **训练/配置页**：
  - 界面提供下拉菜单，允许用户输入 HuggingFace Model ID 替换 I 和 O 分支的大模型底座。

## 4. 技术栈建议
- 训练框架：PyTorch DDP, DeepSpeed Zero-2/3
- 监控记录：W&B (Weights & Biases)
- 后端服务：FastAPI
- 前端框架：React (Next.js) + TailwindCSS
- 流媒体：WebRTC API (浏览器端) + aiortc (Python 端)

## 5. 验收标准
1. DeepSpeed 训练脚本可以成功在至少 2 张 GPU 上启动并完成跑通四个阶段的前反向梯度更新。
2. 前端可以通过页面直接触发模型权重切换并开始一轮 LoRA 微调。
3. WebRTC 通道联通，可从浏览器麦克风说话，并在几百毫秒内听到模型基于 Tri-Transformer 架构生成的语音回包。
