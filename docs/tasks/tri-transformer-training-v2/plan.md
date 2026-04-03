# 任务清单 - Tri-Transformer 训练推进 v2

## 概览

**任务数**: 11（7 TEST + 4 CODE）  
**覆盖 AC**: AC1~AC6 全覆盖  
**执行顺序**: RED（写测试）→ GREEN（写代码）

## TEST 任务（RED 阶段）

| ID | 标题 | 测试文件 |
|----|------|---------|
| T1-1 | CheckpointManager save/load 往返 | tests/test_checkpoint.py |
| T2-1 | CheckpointManager save_best / list_checkpoints | tests/test_checkpoint.py |
| T3-1 | TrainingLogger log/get_history JSONL | tests/test_training_logger.py |
| T4-1 | TrainingLogger 写入失败降级 WARNING | tests/test_training_logger.py |
| T5-1 | Trainer metrics 含 grad_norm | tests/test_trainer_checkpoint.py |
| T6-1 | Trainer train(save_dir) 写 checkpoint | tests/test_trainer_checkpoint.py |
| T7-1 | Trainer train(resume_from) 断点续训 | tests/test_trainer_checkpoint.py |

## CODE 任务（GREEN 阶段）

| ID | 标题 | 依赖 | 文件 |
|----|------|------|------|
| P0-1 | CheckpointManager 实现 | T1-1, T2-1 | checkpoint_manager.py |
| P0-2 | TrainingLogger 实现 | T3-1, T4-1 | training_logger.py |
| P0-3 | Trainer 集成升级 | T5-1~T7-1, P0-1, P0-2 | trainer.py |
| P0-4 | train.py CLI 增强 | P0-3 | scripts/train.py |

## 执行命令

```bash
# 新增测试
cd backend && python -m pytest tests/test_checkpoint.py tests/test_training_logger.py tests/test_trainer_checkpoint.py -v

# 回归测试（验证不破坏现有功能）
cd backend && python -m pytest tests/test_trainer_with_dataloader.py -v

# lint
cd backend && python -m flake8 app/services/train/checkpoint_manager.py app/services/train/training_logger.py app/model/trainer.py scripts/train.py --max-line-length=120
```
