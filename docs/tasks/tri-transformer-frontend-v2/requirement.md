# 需求文档 - Tri-Transformer 前端 v2

## 概述

在 v1 已完成的基础上（54 测试全绿，6 大功能模块完整），v2 迭代目标为：

1. **全双工 WebRTC 实时音视频交互**：浏览器麦克风/摄像头推流 → 后端 Tri-Transformer 实时处理 → 音视频回包
2. **大模型插件配置界面**：支持用户通过 UI 选择 I-Transformer/O-Transformer 底座模型（HuggingFace Model ID）
3. **训练配置与启动**：超参表单 + 阶段选择 + 实时 metrics 轮询

## 背景

来自 `docs/sub_prds/04_training_pipeline_and_full_duplex_frontend.md` 的新增前端需求：

> - 基于 WebRTC 实现麦克风、摄像头的实时推流
> - 画布区域实时显示系统的响应（文字瀑布流、音频播放波形图或视频生成画面）
> - 支持快捷打断按钮和实时风格切换滑块
> - 界面提供下拉菜单，允许用户输入 HuggingFace Model ID 替换 I 和 O 分支的大模型底座

## 功能需求

### F1：WebRTC 实时对话模块（ChatPage 语音 Tab）

| 功能 | 描述 |
|------|------|
| 模态切换 Tab | ChatPage 新增 文本/语音/视频 三个 Tab |
| WebRTC 推流 | getUserMedia 获取 mic/camera，创建 RTCPeerConnection |
| 信令交互 | 与后端交换 SDP offer/answer、ICE candidates |
| 音频波形可视化 | AnalyserNode + requestAnimationFrame 绘制实时波形 |
| 打断按钮 | 发送 interrupt 信号，停止当前输出 |
| 风格切换滑块 | 实时调节「正式度/情绪」参数（通过 data channel 发送） |
| 连接状态显示 | idle / connecting / connected / disconnected / error |

### F2：训练配置界面（TrainingPage 增强）

| 功能 | 描述 |
|------|------|
| 大模型插件选择 | I 端/O 端底座模型 Select（预设常用 + 自定义 HuggingFace Model ID 输入） |
| 超参配置表单 | learning_rate、batch_size、max_steps、训练阶段（0/1/2）|
| 启动训练按钮 | 触发 POST /training/start，展示 jobId |
| 进度轮询 | 每 5s 轮询 GET /training/progress，展示 loss/lr 曲线 |

## 验收标准

| ID | 描述 | 可测试 |
|----|------|--------|
| AC-V2-001 | WebRTC 组件可获取麦克风权限并建立 RTCPeerConnection | ✅ |
| AC-V2-002 | 音频波形可视化组件在有音频流时正确渲染 | ✅ |
| AC-V2-003 | 训练配置表单验证通过后触发 POST /training/start | ✅ |
| AC-V2-004 | 大模型插件配置正确提交 i_model_id/o_model_id | ✅ |
| AC-V2-005 | WebRTC Store 连接状态流转正确 | ✅ |
| AC-V2-006 | 打断按钮触发时发送 interrupt 信号 | ✅ |
| AC-V2-007 | 训练 Store startTraining/fetchProgress 正确 | ✅ |
| AC-V2-008 | ChatPage 模态切换 Tab 功能正常 | ✅ |

## 技术约束

- 不引入新的 npm 包（使用原生 WebRTC API + Web Audio API）
- 测试中用 vitest mock RTCPeerConnection 和 AudioContext
- 复用现有 Zustand、Axios、Ant Design 体系
- 新增代码需通过 TypeScript strict mode
- 已有 54 测试不能退化

## 风险

| ID | 风险 | 缓解 |
|----|------|------|
| RISK-V2-001 | WebRTC jsdom 不支持，Mock 复杂 | 仅测试信令逻辑和状态机，不测底层 WebRTC |
| RISK-V2-002 | AudioContext jsdom 不支持 | Mock AudioContext，测试数据流 |
| RISK-V2-003 | 后端信令服务器未实现 | MSW Mock 信令 API |
