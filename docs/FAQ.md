# 常见问题解答（FAQ）

本文档收集了 Tri-Transformer 用户最常见的问题和解决方案。

---

## 📑 目录

- [安装与部署](#安装与部署)
- [模型与推理](#模型与推理)
- [RAG 知识库](#rag-知识库)
- [性能优化](#性能优化)
- [开发与调试](#开发与调试)
- [错误与故障](#错误与故障)

---

## 安装与部署

### Q1: Docker 启动失败，提示"port is already allocated"

**问题**: 端口被占用

**解决方案**:

```bash
# 检查端口占用
lsof -i :8000
lsof -i :3000

# 修改 docker-compose.yml 中的端口映射
ports:
  - "8001:8000"  # 改为其他端口
  - "3001:3000"
```

### Q2: 后端启动时报"ModuleNotFoundError"

**问题**: Python 依赖未正确安装

**解决方案**:

```bash
cd backend

# 重新安装依赖
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# 验证安装
python -c "import fastapi; import torch; print('OK')"
```

### Q3: 前端构建失败，提示"JavaScript heap out of memory"

**问题**: Node.js 内存不足

**解决方案**:

```bash
# 增加 Node.js 内存限制
export NODE_OPTIONS="--max-old-space-size=4096"
pnpm build

# 或使用更轻量级的构建
pnpm build --mode minimal
```

### Q4: GPU 不可用，PyTorch 提示"CUDA unavailable"

**问题**: CUDA 驱动或 PyTorch 安装问题

**解决方案**:

```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi

# 2. 检查 CUDA 版本
nvcc --version

# 3. 重新安装 PyTorch（匹配 CUDA 版本）
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### Q5: Docker 容器无法访问 GPU

**问题**: NVIDIA Container Toolkit 未配置

**解决方案**:

```bash
# 安装 NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

---

## 模型与推理

### Q6: 模型加载缓慢，需要 10+ 分钟

**问题**: 首次下载模型或磁盘 IO 瓶颈

**解决方案**:

```ini
# .env 配置本地缓存
MODEL_CACHE_DIR=/path/to/fast/ssd/models_cache

# 使用镜像源
HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型后指定本地路径
MODEL_A_NAME=/local/path/to/Qwen2.5-7B
```

### Q7: 推理速度太慢，只有 5-10 tokens/s

**问题**: 未启用优化或批次大小不当

**解决方案**:

```ini
# .env 启用优化
USE_FLASH_ATTENTION=true
USE_VLLM=true
INFERENCE_BATCH_SIZE=32  # 根据显存调整
INFERENCE_DTYPE=bfloat16  # 或 float16
```

```python
# 代码层面验证
from app.model.tri_transformer import TriTransformerConfig

config = TriTransformerConfig()
print(f"Flash Attention: {config.use_flash_attn}")
print(f"Batch Size: {config.inference_batch_size}")
```

### Q8: 模型输出质量差，经常产生幻觉

**问题**: RAG 检索不准确或未启用事实核查

**解决方案**:

```ini
# .env 增强 RAG
RAG_TOP_K=10  # 增加检索文档数
RAG_RERANK=true  # 启用重排序
ENABLE_FACT_CHECK=true  # 启用事实核查
HALLUCINATION_THRESHOLD=0.3  # 降低幻觉检测阈值
```

```bash
# 重新索引文档（提高检索质量）
docker-compose exec backend python -m app.services.rag.reindex
```

### Q9: 如何切换不同的骨干模型？

**方案**:

```ini
# .env 配置
# 使用 Qwen3-8B
MODEL_A_NAME=Qwen/Qwen3-8B
MODEL_B_NAME=Qwen/Qwen3-8B

# 使用 Qwen3-30B-A3B MoE（单卡方案）
MODEL_A_NAME=Qwen/Qwen3-30B-A3B
MODEL_B_NAME=Qwen/Qwen3-30B-A3B
USE_MOE=true

# 使用本地模型
MODEL_A_NAME=/models/Qwen3-8B-Instruct
MODEL_B_NAME=/models/Qwen3-8B-Instruct
```

重启后端生效：

```bash
docker-compose restart backend
```

### Q10: Thinking Mode 是什么？如何使用？

**说明**: Thinking Mode 让模型在生成前进行内部推理，提高复杂问题准确性。

```ini
# .env 配置
ENABLE_THINKING=true  # 全局启用
THINKING_MAX_TOKENS=1024  # 最大推理 token 数
```

API 调用时动态控制：

```json
{
  "message": "复杂数学问题...",
  "enable_thinking": true,
  "thinking_budget": 2048
}
```

---

## RAG 知识库

### Q11: 文档上传后无法检索到内容

**问题**: 文档处理失败或索引未完成

**解决方案**:

```bash
# 1. 检查文档状态
curl -X GET "http://localhost:8000/api/v1/knowledge/documents" \
  -H "Authorization: Bearer $TOKEN"

# 2. 查看处理日志
docker-compose logs backend | grep "document_processor"

# 3. 手动重新处理
docker-compose exec backend \
  python -m app.services.rag.document_processor --reprocess --doc-id=xxx
```

### Q12: 向量数据库连接失败

**问题**: ChromaDB/Milvus 未启动或配置错误

**解决方案**:

```bash
# ChromaDB 方案
docker-compose ps chromadb
docker-compose logs chromadb

# Milvus 方案
docker-compose ps milvus
docker-compose logs milvus

# 重启向量数据库
docker-compose restart chromadb  # 或 milvus
```

### Q13: 如何清空知识库重新索引？

**方案**:

```bash
# ChromaDB
docker-compose exec chromadb rm -rf /chroma/chroma_db

# Milvus
docker-compose exec milvus rm -rf /var/lib/milvus/data

# 重启服务
docker-compose restart chromadb  # 或 milvus

# 重新上传文档
# （通过前端界面或 API）
```

### Q14: RAG 检索结果不相关

**问题**: 嵌入模型不匹配或检索策略不当

**解决方案**:

```ini
# .env 优化检索
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # 更强的模型
RAG_TOP_K=10  # 增加候选数
RAG_USE_HYDE=true  # 启用假设性文档嵌入
RAG_RERANK=true  # 启用 BM25 重排序
RAG_RERANK_WEIGHT=0.3  # 向量相似度权重
```

### Q15: 支持哪些文档格式？

**支持的格式**:

- ✅ PDF（文本型）
- ✅ TXT / MD
- ✅ DOCX / DOC
- ✅ PPTX / PPT
- ✅ XLSX / XLS
- ✅ HTML
- ✅ CSV

**不支持的格式**:

- ❌ 扫描版 PDF（需 OCR 预处理）
- ❌ 加密文档
- ❌ 视频/音频文件（需转录为文本）

---

## 性能优化

### Q16: 如何降低推理延迟？

**优化方案**:

```ini
# .env 配置
USE_VLLM=true  # 使用 vLLM 推理引擎
USE_FLASH_ATTENTION=true  # FlashAttention-3
INFERENCE_DEVICE=cuda  # GPU 加速
INFERENCE_DTYPE=bfloat16  # 混合精度
MAX_SEQ_LEN=2048  # 限制序列长度
ENABLE_THINKING=false  # 关闭思考模式（实时对话）
```

```bash
# 部署优化
docker-compose up -d --scale backend=2  # 多实例负载均衡
```

### Q17: 显存不足，OOM 错误

**问题**: 模型太大或批次大小过大

**解决方案**:

```ini
# .env 减小显存占用
INFERENCE_BATCH_SIZE=8  # 减小批次
MAX_SEQ_LEN=1024  # 缩短序列
USE_QLORA=true  # 4bit 量化（如支持）
GPU_MEMORY_FRACTION=0.8  # 限制显存使用比例
```

```python
# 代码层面优化
import torch
torch.cuda.empty_cache()  # 清理缓存
```

### Q18: 如何提高吞吐量（tokens/s）？

**优化方案**:

1. **启用连续批处理**:

```ini
ENABLE_CONTINUOUS_BATCHING=true
MAX_BATCH_SIZE=64
```

2. **使用 vLLM**:

```ini
USE_VLLM=true
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_SEQS=256
```

3. **多 GPU 并行**:

```ini
TENSOR_PARALLEL_SIZE=2  # 2 卡并行
PIPELINE_PARALLEL_SIZE=1
```

### Q19: CPU 模式下的性能优化

**方案**:

```ini
# .env CPU 优化
INFERENCE_DEVICE=cpu
INFERENCE_DTYPE=float32
USE_OPENVINO=true  # Intel OpenVINO 加速
NUM_THREADS=16  # 线程数
INFERENCE_BATCH_SIZE=4  # 小批次
```

```bash
# 安装 OpenVINO
pip install optimum-intel openvino
```

---

## 开发与调试

### Q20: 如何启用调试模式？

**方案**:

```ini
# .env
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_DEV_TOOLS=true
```

前端：

```ini
# .env.local
VITE_ENABLE_DEBUG=true
VITE_ENABLE_MOCK=false
```

### Q21: 如何查看详细的 API 日志？

**方案**:

```bash
# 后端详细日志
docker-compose logs -f backend | grep "API"

# 或进入容器查看
docker-compose exec backend tail -f logs/app.log
```

### Q22: 如何 Mock API 进行前端开发？

**方案**:

```ini
# .env.local
VITE_ENABLE_MOCK=true
```

编辑 Mock 数据：

```typescript
// frontend/src/mocks/handlers/chat.ts
export const chatHandlers = [
  rest.post('/api/v1/chat/sessions', (req, res, ctx) => {
    return res(ctx.json({
      id: 'mock-session-id',
      created_at: new Date().toISOString(),
    }))
  }),
]
```

### Q23: 如何运行单个测试？

**方案**:

```bash
# 后端单个测试
cd backend
pytest tests/test_chat.py::test_create_session -v

# 前端单个测试
cd frontend
pnpm test -- ChatInput.test.tsx

# 覆盖率报告
pytest --cov=app tests/test_chat.py
```

---

## 错误与故障

### Q24: "RuntimeError: CUDA out of memory"

**解决方案**:

```ini
# .env
INFERENCE_BATCH_SIZE=4
MAX_SEQ_LEN=512
GPU_MEMORY_FRACTION=0.5
```

```bash
# 清理显存
docker-compose restart backend

# 或手动清理
docker-compose exec backend python -c "import torch; torch.cuda.empty_cache()"
```

### Q25: "ConnectionRefusedError: [Errno 111] Connection refused"

**问题**: 服务未启动或端口错误

**解决方案**:

```bash
# 检查服务状态
docker-compose ps

# 查看网络配置
docker-compose exec backend ping localhost

# 检查防火墙
sudo ufw status
sudo ufw allow 8000
```

### Q26: "KeyError: 'access_token'"

**问题**: JWT 认证失败

**解决方案**:

```bash
# 重新登录获取 Token
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin123!@#"}' \
  | jq -r '.access_token')

# 验证 Token
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

### Q27: 前端页面空白，控制台报错"CORS policy"

**问题**: CORS 配置不当

**解决方案**:

```ini
# backend/.env
FRONTEND_URL=http://localhost:3000
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

重启后端：

```bash
docker-compose restart backend
```

### Q28: "sqlite3.OperationalError: database is locked"

**问题**: SQLite 数据库锁死

**解决方案**:

```bash
# 停止服务
docker-compose down

# 删除数据库文件
rm backend/tri_transformer.db

# 重新启动
docker-compose up -d
```

---

## 获取帮助

### 仍未解决您的问题？

1. **搜索 Issue**: [GitHub Issues](https://github.com/your-org/tri-transformer/issues)
2. **提交 Issue**: 提供详细错误信息和复现步骤
3. **查看日志**: `docker-compose logs -f`
4. **社区讨论**: [Discussions](https://github.com/your-org/tri-transformer/discussions)

### 提交 Issue 模板

```markdown
**问题描述**:
简要描述问题

**复现步骤**:
1. ...
2. ...
3. ...

**环境信息**:
- OS: Ubuntu 22.04
- Python: 3.10
- Docker: 24.0
- GPU: RTX 4090

**日志**:
```
粘贴相关日志
```

**期望行为**:
描述期望的结果
```

---

**最后更新**: 2026-04-03  
**维护者**: Tri-Transformer Team
