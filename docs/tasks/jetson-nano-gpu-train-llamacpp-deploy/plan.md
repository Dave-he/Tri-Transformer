# 任务清单：Jetson Nano 8GB GPU 训练 + llama.cpp 部署

**Task ID**: jetson-nano-gpu-train-llamacpp-deploy  
**版本**: v1.0  
**日期**: 2026-04-28

---

## 任务概览（9 个任务）

| ID | 优先级 | 标题 | 关键文件 |
|----|--------|------|---------|
| T1 | P0 | Jetson Nano 硬件检测模块 | jetson_device.py |
| T2 | P0 | TrainerConfig Jetson Nano 适配 | trainer.py |
| T3 | P0 | Settings 配置扩展 | config.py |
| T4 | P0 | GGUF 转换器 | gguf_converter.py |
| T5 | P1 | llama.cpp 推理服务封装 | llama_cpp_service.py |
| T6 | P1 | 推理模式切换 API | model.py |
| T7 | P1 | 训练配置文件 + CLI 适配 | jetson_nano_config.yaml, train.py |
| T8 | P1 | Jetson Nano 安装脚本 | install_jetson_deps.sh, build_llamacpp_jetson.sh |
| T9 | P1 | 文档更新 | PRD.md, architecture.md, 新增2篇指南 |

---

## T1: Jetson Nano 硬件检测模块 (P0)

- 新增 `backend/app/model/jetson_device.py`
- `JetsonDeviceInfo` 数据类：is_jetson, gpu_name, cuda_version, total_memory_gb, is_shared_memory, cuda_cores
- `detect_jetson_device()`: 读取 /etc/nv_tegra_release, nvidia-smi, /proc/meminfo
- `get_memory_usage_pct()`: 共享内存使用百分比
- 测试：mock Jetson 环境 + 非 Jetson 环境

## T2: TrainerConfig Jetson Nano 适配 (P0)

- `TrainerConfig` 新增 `jetson_nano: bool = False`
- Jetson 模式下 GradScaler(init_scale=1024)
- train() 中集成内存监控 WARNING

## T3: Settings 配置扩展 (P0)

- `inference_mode: str = "pytorch_direct"` (可选 llamacpp_gguf)
- `llamacpp_model_path: Optional[str] = None`
- `jetson_memory_limit_pct: float = 0.85`

## T4: GGUF 转换器 (P0)

- 新增 `backend/app/model/gguf_converter.py`
- 仅转换 O-Transformer Streaming Decoder 分支
- 流程：checkpoint.pt → safetensors → F16 GGUF → 量化
- 支持量化选项 Q4_K_M, Q5_K_M, Q8_0

## T5: llama.cpp 推理服务封装 (P1)

- 新增 `backend/app/services/inference/llama_cpp_service.py`
- 封装 llama-cpp-python Llama 类
- generate() / chat() 方法
- 可选依赖，缺失时 ImportError

## T6: 推理模式切换 API (P1)

- 修改 `backend/app/api/v1/model.py`
- GET /model/inference-mode + POST /model/inference-mode

## T7: 训练配置文件 + CLI 适配 (P1)

- 新增 `backend/configs/jetson_nano_config.yaml`
- train.py 新增 --jetson-nano, --galore-rank
- train_service.py GaLore defaults 适配 Jetson (rank=64)

## T8: Jetson Nano 安装脚本 (P1)

- `install_jetson_deps.sh`: JetPack PyTorch + llama.cpp + llama-cpp-python
- `build_llamacpp_jetson.sh`: llama.cpp 编译（含 6 处 patch）
- `convert_to_gguf.py`: CLI GGUF 转换入口

## T9: 文档更新 (P1)

- PRD.md 新增 §4.7.2 Jetson Nano 部署方案
- architecture.md 新增 Jetson 架构图
- 新增 jetson_nano_guide.md（环境搭建/训练限制/优化策略）
- 新增 llamacpp_deployment.md（转换/量化/部署/集成）
- training_guide.md 新增 Jetson 训练章节
