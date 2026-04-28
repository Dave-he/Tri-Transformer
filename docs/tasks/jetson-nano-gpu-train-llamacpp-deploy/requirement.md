# 需求文档：Jetson Nano 8GB GPU 训练 Tri-Transformer 模型 + llama.cpp 部署

**Task ID**: jetson-nano-gpu-train-llamacpp-deploy  
**版本**: v1.0  
**日期**: 2026-04-28  
**平台**: Backend/Python (aarch64 + CUDA 10.2)

---

## 1. 需求背景

Tri-Transformer 三分支模型架构已完整实现（I/C/O 三分支 + adaLN-Zero 控制机制 + LoRA/GaLore 训练支持），当前训练脚本使用 dummy 数据和标准 GPU 环境运行。

现需将训练和部署适配到 **NVIDIA Jetson Nano 8GB** 边缘设备上，实现：

- **本地训练**：在 Jetson Nano 的 Maxwell GPU 上训练 Tri-Transformer 轻量配置模型
- **llama.cpp 部署**：训练完成后，使用 llama.cpp 的 GGUF 格式进行推理部署

## 2. 硬件环境约束

### Jetson Nano 8GB 关键限制

| 约束 | 详情 | 影响 |
|------|------|------|
| GPU 架构 | Maxwell GM20B, sm_53 | 无 FlashAttention/DeepSpeed/vLLM |
| CUDA 版本 | 10.2 only（不可升级） | PyTorch <= 1.13.x, 无 torch.compile |
| 显存 | 8GB LPDDR4 **共享 CPU+GPU** | 约 6GB 可用于 ML, 无独立 VRAM |
| CUDA Cores | 128 | 训练速度慢, 需 gradient accumulation |
| Tensor Cores | 无（Maxwell 缺乏） | AMP 仅节省内存, 无速度提升 |
| CPU | 4-core ARM Cortex-A57 @ 1.43GHz | aarch64, 多数预编译库不支持 |
| 带宽 | ~25.6 GB/s | 数据传输慢 |

### 可行模型配置

| 配置 | 参数量 | FP16 模型 | GaLore 训练 | 部署可行? |
|------|--------|----------|------------|----------|
| 默认轻量 (d=512, vocab=151936) | ~325M | 619 MB | ~1.7 GB | ✅ 舒适 |
| 紧凑 (d=512, vocab=32000) | ~140M | 268 MB | ~1.1 GB | ✅ 更轻松 |
| 迷你 (d=256, vocab=8000) | ~29M | 55 MB | ~220 MB | ✅ 轻量 |
| QWEN3_8B (d=4096) | ~8.8B | 16.4 GB | — | ❌ 不可能 |

## 3. 功能需求

### FR1: Jetson Nano 训练适配

- 自动检测 Jetson Nano 硬件环境（GPU型号、内存、CUDA版本）
- 检测到 Jetson Nano 时，自动禁用 FlashAttention/DeepSpeed
- 训练配置：`batch_size=1, gradient_accumulation_steps=4, GaLoreAdamW(rank=64), AMP(FP16)`
- GradScaler init_scale=1024（Maxwell 无 Tensor Cores，需保守缩放）
- 实时监控共享内存使用率，超 85% WARNING

### FR2: 轻量训练配置

- 提供 `jetson_nano_config.yaml` 预置配置
- 默认配置 ~325M 参数，GaLore 训练 ~1.7GB 内存占用
- 三阶段训练：Stage1 LoRA(I+O)+冻结C → Stage2 LoRA(C)+冻结I+O → Stage3 全量（可选）

### FR3: PyTorch → GGUF 转换

- 提供 `convert_to_gguf.py` 转换脚本
- 转换流程：`.pt → safetensors → GGUF F16 → 量化(Q4_K_M/Q5_K_M/Q8_0)`
- Tri-Transformer 自定义架构 GGUF 元数据注册
- 325M 模型 Q5_K_M 量化后约 1.8GB

### FR4: llama.cpp 推理服务部署

- llama.cpp server 在 aarch64 + CUDA 10.2 上编译运行（需 6 处源码 patch）
- 提供 llama_cpp_server.py FastAPI 集成模块
- 两种推理路径：
  - **PyTorch 直接推理**：支持三分支全功能（幻觉检测、RAG、实时打断）
  - **llama.cpp GGUF 推理**：仅 O-Transformer 输出分支（轻量部署场景）
- 推理模式通过配置切换

### FR5: Jetson Nano 环境安装脚本

- `install_jetson_deps.sh` 一键安装 JetPack PyTorch + llama.cpp + llama-cpp-python
- 安装后验证：`torch.cuda.is_available()` = True, `llama-server --version` 正常

### FR6: 文档更新

- PRD.md 新增 Jetson Nano 部署章节
- architecture.md 新增 Jetson Nano 架构图
- 新增 jetson_nano_guide.md 和 llamacpp_deployment.md

## 4. 不在范围内

- QWEN3-8B 全量骨干训练（需要 16+GB）
- FlashAttention / DeepSpeed / vLLM（Maxwell 不支持）
- 多 GPU 分布式训练
- 音频/视觉模态训练
- Tri-Transformer 三分支全架构 llama.cpp C++ 推理实现

## 5. 部署架构

```
[Jetson Nano 8GB]
    ├── FastAPI Backend (CPU)
    │     ├── Tri-Transformer PyTorch 推理 (FP16, ~619MB)
    │     ├── llama.cpp 推理 (GGUF Q5_K_M, ~1.8GB)
    │     ├── RAG pipeline (ChromaDB + sentence-transformers)
    │     └── 推理模式切换 (pytorch_direct / llamacpp_gguf)
    ├── Training (GPU + CPU, 共享内存)
    │     ├── Stage 1: LoRA(I+O), freeze C
    │     ├── Stage 2: LoRA(C), freeze I+O
    │     ├── GaLoreAdamW + AMP(FP16) + grad_accum=4
    └── Frontend (nginx 静态文件, 另机器或本地)
```
