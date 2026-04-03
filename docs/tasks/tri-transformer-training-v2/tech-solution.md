# 技术方案 - Tri-Transformer 训练推进 v2

## 背景与目标

在已完成的三分支模型（136测试全通过，dummy训练可运行）基础上，增加：
1. **CheckpointManager**：断点续训能力
2. **TrainingLogger**：结构化指标监控
3. **CLI 增强**：`--save-dir / --resume / --log-file`

## 架构设计

```
backend/app/services/train/
├── checkpoint_manager.py  [NEW]
├── training_logger.py     [NEW]
└── dataset_loader.py      [EXISTING]

backend/app/model/
└── trainer.py             [MODIFY: +grad_norm, +save_dir, +resume_from]

backend/scripts/
└── train.py               [MODIFY: +--save-dir, +--resume, +--log-file]
```

## CheckpointManager

```python
class CheckpointManager:
    def save(self, trainer, path: str) -> None:
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'epoch': current_epoch,
            'loss': current_loss,
        }, path)

    def load(self, trainer, path: str) -> int:  # 返回 epoch
        ...

    def save_best(self, trainer, save_dir: str, metric: float) -> bool:
        ...

    def list_checkpoints(self, save_dir: str) -> list[str]:
        ...  # 按 epoch 数字排序
```

## TrainingLogger

```python
class TrainingLogger:
    def __init__(self, log_file: str | None = None):
        self._history = []
        self._log_file = log_file

    def log(self, metrics: dict) -> None:
        self._history.append(metrics)
        if self._log_file:
            try:
                with open(self._log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
            except Exception as e:
                warnings.warn(f"TrainingLogger write failed: {e}")

    def get_history(self) -> list[dict]:
        return list(self._history)
```

## Trainer 集成

`train()` 签名升级：
```python
def train(
    self,
    data_loader=None,
    max_steps: int = None,
    save_dir: str = None,        # [NEW]
    resume_from: str = None,     # [NEW]
    log_file: str = None,        # [NEW]
) -> list[dict]:
```

新增 `grad_norm` 计算（在 `clip_grad_norm_` 时捕获）：
```python
grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
```

## 数据流

```
train() 每 epoch：
  compute_loss → backward → clip_grad(→ grad_norm) → step
  → metrics = {epoch, loss, lr, grad_norm, stage, progress}
  → logger.log(metrics)
  → ckpt_mgr.save(f"{save_dir}/epoch_{epoch:03d}.pt")
  → ckpt_mgr.save_best(save_dir, loss)
```

## 风险

| 风险 | 缓解 |
|------|------|
| trainer.py 改动破坏现有测试 | 所有新参数默认 None，完全向后兼容 |
| 磁盘写入失败中断训练 | logger 写失败仅 WARNING |

## 验收标准

见 tech-solution.yaml → validation.acceptance
