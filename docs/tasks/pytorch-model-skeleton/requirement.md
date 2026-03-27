# 需求文档：搭建 Tri-Transformer 模型骨架并集成 PyTorch 训练

## 背景

项目已有完整的 FastAPI 后端基础架构（用户认证、RAG 知识库、训练任务调度 API）。当前 `TrainService` 仅做任务入库，无实际 PyTorch 模型骨架与训练逻辑。

本次需求：在服务端新增 **Tri-Transformer 三分支 PyTorch 模型骨架** 及 **训练器（Trainer）**，并将训练流程通过 FastAPI BackgroundTasks 异步驱动，训练指标写回数据库。

---

## 功能需求

### FR-01: Tri-Transformer 模型骨架

实现三分支 PyTorch 模型（`backend/app/services/model/tri_transformer.py`）：

| 分支 | 实现 | 职责 |
|------|------|------|
| **ITransformer** | TransformerEncoder + 位置编码 | 输入编码，接受控制信号 |
| **CTransformer** | 双向交叉注意力 MultiheadAttention | 全局控制中枢，生成控制信号 |
| **OTransformer** | 带因果掩码 TransformerDecoder | 受控输出生成 |
| **TriTransformerModel** | 串联三分支 | 对外 forward() |

**验收标准：**
- 任意 `(batch, seq_len)` 的 Tensor 可完成一次前向传播
- 输出 logits 形状为 `(batch, tgt_seq_len, vocab_size)`
- C 分支控制信号正确注入 I/O 分支

### FR-02: PyTorch 训练器

实现 `TriTransformerTrainer`（`backend/app/services/train/pytorch_trainer.py`）：

- 三种训练阶段：
  - **stage1**：LoRA 基础微调，冻结 CTransformer 参数
  - **stage2**：控制中枢训练，冻结 ITransformer + OTransformer
  - **stage3**：RAG 适配全量训练
- AdamW 优化器 + 余弦退火调度
- 每 epoch 通过 `metrics_callback` 回调更新 loss/epoch/progress
- 支持 `cancel_event`（`threading.Event`）中止训练

**验收标准：**
- stage1 冻结 CTransformer 所有参数
- stage2 冻结 ITransformer + OTransformer 所有参数
- cancel_event 置位后训练停止
- metrics_callback 每 epoch 被调用，携带 `{loss, epoch, total_epochs}` 信息

### FR-03: 训练任务后台执行集成

扩展 `TrainService`，在 `submit_job` 中启动 BackgroundTask：

- `POST /train/jobs` 后立即启动后台训练
- 后台任务：`pending → running → completed/failed`
- 每 epoch metrics 写入 `job.config` 的 `metrics` 字段
- `DELETE /train/jobs/{id}` 发送 cancel 信号中止训练

**验收标准：**
- 提交任务后 job.status 最终变为 completed 或 failed
- job.config 的 metrics 字段记录各 epoch loss
- 调用 DELETE 后 job.status 变为 cancelled

### FR-04: 配置扩展

在 `Settings` 中新增训练超参数配置：

```python
d_model: int = 256
num_heads: int = 8
num_layers: int = 6
vocab_size: int = 32000
max_seq_len: int = 512
train_epochs_default: int = 3
train_lr_default: float = 1e-4
train_device: str = "cpu"
```

---

## 非功能需求

- 模型骨架仅依赖 `torch`，不引入 `transformers` 库
- CPU 环境可运行（`TRAIN_DEVICE=cpu`）
- 新增代码有对应 pytest 单元测试
- 不破坏现有测试

---

## 范围外

- 真实语料数据集加载
- LoRA/PEFT 微调
- 模型 checkpoint 持久化
- 分布式训练
- GPU 推理集成
