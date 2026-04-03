# 需求文档 - Tri-Transformer 训练推进 v2

## 背景

Tri-Transformer 三分支模型架构已完整（Qwen3 风格 GQA+RoPE+adaLN-Zero），136 测试全通过，
`train.py --dataset dummy` 可正常运行。本轮推进三个核心能力：

1. **Checkpoint 管理**：自动保存/恢复，支持断点续训
2. **训练监控**：结构化 JSON Lines 日志记录
3. **CLI 增强**：`--resume` / `--save-dir` / `--log-file` 参数

## 功能需求

### FR1: CheckpointManager

```python
class CheckpointManager:
    def save(trainer, path: str) -> None
    def load(trainer, path: str) -> int  # 返回恢复的 epoch
    def save_best(trainer, save_dir: str, metric: float) -> bool
    def list_checkpoints(save_dir: str) -> list[str]  # 按 epoch 排序
```

### FR2: TrainingLogger

```python
class TrainingLogger:
    def __init__(log_file: str | None = None)
    def log(metrics: dict) -> None   # 追加写入 JSONL
    def get_history() -> list[dict]  # 内存读取
```

### FR3: Trainer 集成

- `train(save_dir=None, resume_from=None)` 新增参数
- 每 epoch 末调用 `CheckpointManager.save()`（save_dir 非 None）
- metrics 中增加 `grad_norm` 字段

### FR4: CLI 增强

```
python train.py --dataset dummy --epochs 3 --save-dir ./checkpoints --log-file train.jsonl
python train.py --dataset dummy --epochs 5 --resume ./checkpoints/epoch_002.pt
```

## 验收标准

| AC | 描述 |
|----|------|
| AC1 | CheckpointManager.save/load 往返一致，epoch 正确恢复 |
| AC2 | TrainingLogger 写入 JSONL，get_history() 正确读取 |
| AC3 | Trainer.train(save_dir=...) 触发 checkpoint 写入 |
| AC4 | Trainer metrics 包含 grad_norm 字段 |
| AC5 | train.py --resume PATH 输出 "Resuming from epoch X" |
| AC6 | 所有新增 pytest 通过 |

## 范围外

- 真实网络下载 ModelScope 数据集
- 分布式/混合精度训练
