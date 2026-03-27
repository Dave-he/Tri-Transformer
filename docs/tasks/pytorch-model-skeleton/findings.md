# 研究发现

## 背景
Tri-Transformer 后端已有完整 FastAPI 架构，包括用户认证、RAG 知识库、训练任务 CRUD API。需在此基础上加入实际的 PyTorch 模型骨架和训练逻辑。

## 关键发现

### 发现 1: 现有训练服务仅做 DB CRUD
- **来源**: backend/app/services/train/train_service.py
- **内容**: TrainService 只有 submit_job/get_job/cancel_job/list_jobs，无任何训练逻辑
- **影响**: 需要新增 pytorch_trainer.py，并在 submit_job 或 API 层集成 BackgroundTasks

### 发现 2: BackgroundTasks 集成方式
- **来源**: backend/app/api/v1/train.py
- **内容**: submit_job 路由接收 BackgroundTasks 参数后可立即 add_task
- **影响**: 训练后台任务需要独立 DB session（不能复用请求 session）

### 发现 3: 现有 requirements.txt 无 torch
- **来源**: backend/requirements.txt
- **内容**: 无 torch 依赖，仅有 sentence-transformers（内部依赖 torch）
- **影响**: 需要显式添加 torch>=2.0.0，建议 CPU 版本

### 发现 4: 测试框架为 pytest-asyncio（asyncio_mode=auto）
- **来源**: backend/pyproject.toml
- **内容**: pytest-asyncio 已配置，SQLite in-memory 测试 DB
- **影响**: 新增测试需遵循现有 conftest.py fixtures 模式

### 发现 5: Settings 已有模型路径配置，需扩展
- **来源**: backend/app/core/config.py
- **内容**: 已有 model_path/embedding_model_path，需新增 d_model/num_heads 等超参数
- **影响**: TriTransformerModel 初始化从 settings 读取超参数

## 技术笔记
- 三分支架构：I(Encoder) → C(CrossAttn 双向) → O(Decoder with causal mask)
- C 分支对 I 分支编码输出做交叉注意力，生成控制向量
- O 分支在 decoder 的 memory 位置融合 I 编码，在 cross-attn 位置融合 C 控制信号
- 测试中使用 d_model=32, num_heads=2, num_layers=1 以加速
