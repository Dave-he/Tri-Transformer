# 技术方案：Jetson Nano 8GB GPU 训练 + llama.cpp 部署

**Task ID**: jetson-nano-gpu-train-llamacpp-deploy  
**版本**: v1.0  
**风险级别**: MEDIUM  
**日期**: 2026-04-28

---

## 1. 方案概述

适配 Jetson Nano 8GB（Maxwell GPU, CUDA 10.2, 8GB 共享内存）训练 Tri-Transformer 轻量配置模型，使用 llama.cpp GGUF 量化格式部署推理服务。

**核心策略**：

| 环节 | 方案 | 理由 |
|------|------|------|
| 训练配置 | 默认轻量 (d=512, vocab=151936, ~325M) | GaLore训练 ~1.7GB, 舒适适配 8GB |
| 训练优化 | GaLoreAdamW + AMP(FP16) + grad_accum=4 | 内存高效, 无 Tensor Core 速度损失 |
| GGUF 转换 | 仅 O-Transformer Streaming Decoder | 标准 causal decoder, llama.cpp 可原生转换 |
| 推理部署 | PyTorch 直接推理为主, llama.cpp GGUF 为轻量备选 | 三分支全功能 vs 单分支轻量 |

## 2. 系统架构

```
[Jetson Nano 8GB]
    ├── FastAPI Backend (CPU)
    │     ├── InferenceService (pytorch_direct)
    │     │     └── TriTransformerModel.forward() → 三分支全功能推理
    │     ├── LlamaCppService (llamacpp_gguf)
    │     │     └── llama-cpp-python Llama(Q5_K_M.gguf) → 轻量单分支推理
    │     ├── 推理模式切换: inference_mode 配置
    │     └── RAG pipeline (ChromaDB + sentence-transformers)
    ├── Training (GPU + CPU, 共享内存)
    │     ├── JetsonDevice.detect() → 硬件适配
    │     ├── TriTransformerTrainer(GaLore+AMP+grad_accum)
    │     ├── MemoryMonitor → 85%阈值 WARNING
    │     └── Checkpoint → GGUF 转换管道
    └── CLI Scripts
        ├── train.py --jetson-nano
        ├── convert_to_gguf.py
        ├── install_jetson_deps.sh
        └── build_llamacpp_jetson.sh
```

## 3. 关键组件设计

### 3.1 Jetson Nano 硬件检测 (jetson_device.py)

```python
@dataclass
class JetsonDeviceInfo:
    is_jetson: bool
    gpu_name: str         # "NVIDIA Jetson Nano"
    cuda_version: str     # "10.2"
    total_memory_gb: float # 8.0
    is_shared_memory: bool # True
    cuda_cores: int       # 128

def detect_jetson_device() -> JetsonDeviceInfo:
    # 读取 /etc/nv_tegra_release 判断 Jetson
    # 读取 nvidia-smi 获取 GPU/CUDA 信息
    # 读取 /proc/meminfo 获取内存信息

def get_memory_usage_pct() -> float:
    # 返回当前 CPU+GPU 共享内存使用百分比
```

### 3.2 GGUF 转换器 (gguf_converter.py)

**仅转换 O-Transformer Streaming Decoder**（I/C 分支结构无法适配 llama.cpp 标准 LLM 转换）：

```
checkpoint.pt → 提取 o_branch.streaming_decoder 权重
    → 保存为 HuggingFace safetensors 格式
    → 注册自定义 GGUF 元数据 (tri_transformer.o_transformer 标记)
    → convert_hf_to_gguf.py → F16 GGUF
    → llama-quantize → Q4_K_M / Q5_K_M / Q8_0
```

O-Transformer Streaming Decoder 结构兼容 llama.cpp：
- 标准 causal decoder（逐层 DecoderLayer）
- GQA (num_kv_heads < num_heads) → llama.cpp 已支持
- RoPE → llama.cpp 已支持
- SwiGLU → llama.cpp 已支持

### 3.3 llama.cpp 推理服务 (llama_cpp_service.py)

```python
class LlamaCppService:
    def __init__(self, model_path: str, n_gpu_layers: int = 0):
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers)
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        output = self.llm(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"]
    
    def chat(self, messages: list) -> str:
        output = self.llm.create_chat_completion(messages=messages)
        return output["choices"][0]["message"]["content"]
```

### 3.4 训练适配

| 项目 | 标准配置 | Jetson Nano 适配 |
|------|----------|-----------------|
| batch_size | 8 | 1 |
| grad_accum | 4 | 4 (有效 batch=4) |
| GaLore rank | 128 | 64 |
| GradScaler init | default | 1024 (Maxwell 保守) |
| FlashAttention | 可选 | 禁用 (Maxwell 不支持) |
| 内存监控 | 无 | 85% 阈值 WARNING |

### 3.5 llama.cpp Jetson Nano 编译

需 6 处源码 patch：
1. `constexpr` → `const`（aarch64 兼容）
2. `__builtin_assume` → 替代实现
3. bfloat16 类型兼容
4. CUDA arch 目标设为 `50 61`
5. `cmath` 头文件兼容
6. `unistd.h` ARM 兼容

编译时间约 85 分钟（Jetson Nano CPU）。

## 4. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| backend/app/model/jetson_device.py | 新增 | Jetson 硬件检测 + 内存监控 |
| backend/app/model/gguf_converter.py | 新增 | O-Transformer → GGUF 转换 |
| backend/app/model/trainer.py | 修改 | Jetson 训练适配 |
| backend/app/core/config.py | 修改 | 新增推理模式/llama.cpp/Jetson 配置 |
| backend/app/services/inference/llama_cpp_service.py | 新增 | llama.cpp 推理封装 |
| backend/app/services/train/train_service.py | 修改 | GaLore 适配 Jetson |
| backend/app/api/v1/model.py | 修改 | 推理模式切换 |
| backend/scripts/train.py | 修改 | --jetson-nano 参数 |
| backend/scripts/install_jetson_deps.sh | 新增 | 一键安装 |
| backend/scripts/convert_to_gguf.py | 新增 | CLI GGUF 转换 |
| backend/scripts/build_llamacpp_jetson.sh | 新增 | llama.cpp 编译脚本 |
| backend/configs/jetson_nano_config.yaml | 新增 | 预置训练配置 |
| docs/PRD.md | 修改 | 新增部署章节 |
| docs/agent/architecture.md | 修改 | 新增架构图 |
| docs/agent/jetson_nano_guide.md | 新增 | Jetson 开发指南 |
| docs/agent/llamacpp_deployment.md | 新增 | llama.cpp 部署指南 |
| backend/tests/test_jetson_device.py | 新增 | 硬件检测测试 |
| backend/tests/test_gguf_converter.py | 新增 | GGUF 转换测试 |
| backend/tests/test_llama_cpp_service.py | 新增 | 推理服务测试 |

## 5. 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| llama.cpp aarch64 编译 patch 随版本失效 | MEDIUM | 固定版本, 自动 patch 脚本 |
| GGUF 推理丢失 I/C 幻觉检测能力 | MEDIUM | 默认 pytorch_direct 模式, llamacpp 仅轻量备选 |
| Jetson 8GB 内存溢出 | LOW | GaLore+AMP+85%阈值监控 |
| PyTorch for Jetson 版本限制 | LOW | 代码兼容 PyTorch 1.10+ |
