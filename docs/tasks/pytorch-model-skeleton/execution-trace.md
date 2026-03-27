# Execution Trace - pytorch-model-skeleton

> 需求: 搭建模型骨架，服务端使用 PyTorch 训练模型
> 创建时间: 2026-03-27

## 执行摘要

**任务**: pytorch-model-skeleton
**状态**: DONE
**执行模式**: standard
**开始时间**: 2026-03-27T00:00:00Z
**结束时间**: 2026-03-27

### Stage 执行结果

| Stage | 名称 | 状态 | 产物 |
|-------|------|------|------|
| Stage 0 | 前置检查 | DONE | - |
| Stage 1 | 需求门禁 | DONE | requirement.yaml |
| Stage 1.5 | Figma 解析 | SKIP | 非 UI 项目 |
| Stage 3 | 技术方案 | DONE | branches.py / tri_transformer.py / trainer.py |
| Stage 5 | 任务拆解 | DONE | T1-T9 |
| Stage 7 | TDD 实现 | DONE | 代码 + 测试 80/80 通过 |

---

## Stage 执行记录

### T1 - branches.py（三分支模型骨架）

- `backend/app/model/branches.py`
- `PositionalEncoding`：sinusoidal PE，register_buffer，支持可变长序列
- `ITransformer`：nn.Embedding + PositionalEncoding + TransformerEncoderLayer(norm_first=True) + LayerNorm，支持 `control_signal` 叠加
- `CTransformer`：state_slot 可学习参数（1×1×d_model），多层 self-attn + cross-attn-I + cross-attn-O + FFN，输出形状 (B,1,d_model)
- `OTransformer`：nn.Embedding + PositionalEncoding + TransformerDecoderLayer，ctrl_proj 将控制信号拼接到 memory，自动生成因果 mask，输出 (logits, hidden)

### T2 - tri_transformer.py（主模型）

- `backend/app/model/tri_transformer.py`
- `TriTransformerConfig`：dataclass，含 vocab_size/d_model/num_heads/num_layers_i,c,o/dim_feedforward/dropout/max_len/pad/bos/eos token id
- `TriTransformerOutput`：dataclass，含 logits/i_hidden/control_signal/o_hidden
- `TriTransformerModel`：三分支融合，支持 o_prev 反馈，`_init_weights` Xavier/normal 初始化，`num_parameters(trainable_only)`

### T3 - trainer.py（PyTorch 训练器）

- `backend/app/model/trainer.py`
- `TrainerConfig`：dataclass，job_type/num_epochs/lr/wd/batch_size/seq_len/vocab_size/device/model_config
- `STAGE_MAP`：lora_finetune→stage1, full_finetune→stage2, rag_adapt/dpo_align→stage3
- `TriTransformerTrainer`：
  - `_freeze_by_stage`：stage1 冻结 C, stage2 冻结 I+O, stage3 全量
  - `_make_dummy_batch`：生成随机 src/tgt_in/tgt_out
  - `train_epoch`：前向+交叉熵+反向+梯度裁剪+optimizer+scheduler
  - `train`：支持 cancel_event 中断、metrics_callback 回调、返回 history 列表
  - AdamW + CosineAnnealingLR，CrossEntropyLoss(ignore_index=0)

### T4 - config.py 扩展

- 新增 model_hidden_dim/model_num_heads/model_num_layers/model_dropout/train_epochs/train_batch_size/train_lr/checkpoint_dir

### T5 - train.py 集成

- `backend/app/api/v1/train.py`
- `POST /train/jobs/{job_id}/start` 通过 BackgroundTasks 触发 TriTransformerTrainer
- metrics_callback 实时更新 job.metrics，cancel_event 支持停止训练

### T6 - test_model_skeleton.py（34 个测试）

- `backend/tests/test_model_skeleton.py`
- TestPositionalEncoding (2)、TestITransformer (4)、TestCTransformer (4)、TestOTransformer (3)
- TestTriTransformerModel (8)、TestTriTransformerConfig (2)、TestStageMap (2)
- TestTriTransformerTrainer (9)：stage 冻结、训练循环、cancel 中断、callback、progress

### T7 - 全量测试验证

```
80 passed in 49.14s
```

- 原有 46 个 API 测试全部保留通过
- 新增 34 个模型骨架测试全部通过

---

## 反思分析

- CTransformer 使用 state_slot 可学习参数而非输入 embedding，使其真正成为"全局控制中枢"
- norm_first=True（Pre-LN）提升深层训练稳定性
- 三阶段冻结策略（stage1/2/3）分别针对基础对话能力、控制对齐、RAG/偏好对齐，逐步解冻
- cancel_event + threading.Event 支持 FastAPI BackgroundTasks 中途停止训练
