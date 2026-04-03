import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.model.tri_transformer import TriTransformerModel, TriTransformerConfig


@dataclass
class TrainerConfig:
    job_type: str = "lora_finetune"
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    seq_len: int = 64
    vocab_size: int = 32000
    device: str = "cpu"
    model_config: Optional[TriTransformerConfig] = None


STAGE_MAP = {
    "lora_finetune": "stage1",
    "full_finetune": "stage2",
    "rag_adapt": "stage3",
    "dpo_align": "stage3",
    "stage1": "stage1",
    "stage2": "stage2",
    "stage3": "stage3",
}


class TriTransformerTrainer:
    """
    三分支 Transformer 训练器，支持三个训练阶段：

    Stage 1 (lora_finetune): 冻结 CTransformer，训练 I/O 基础对话能力。
    Stage 2 (full_finetune): 冻结 ITransformer + OTransformer，专项训练 CTransformer 控制对齐。
    Stage 3 (rag_adapt/dpo_align): 全量微调，RAG 适配与偏好对齐。
    """

    def __init__(
        self,
        config: TrainerConfig,
        metrics_callback: Optional[Callable[[dict], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.metrics_callback = metrics_callback
        self.cancel_event = cancel_event or threading.Event()

        model_cfg = config.model_config or TriTransformerConfig(
            vocab_size=config.vocab_size,
        )
        self.model = TriTransformerModel(model_cfg).to(config.device)
        self.stage = STAGE_MAP.get(config.job_type, "stage1")
        self._freeze_by_stage()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 1e-2,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _freeze_by_stage(self):
        for p in self.model.parameters():
            p.requires_grad = True

        if self.stage == "stage1":
            for p in self.model.c_branch.parameters():
                p.requires_grad = False
        elif self.stage == "stage2":
            for p in self.model.i_branch.parameters():
                p.requires_grad = False
            for p in self.model.o_branch.parameters():
                p.requires_grad = False

    def _make_dummy_batch(self) -> tuple:
        B = self.config.batch_size
        S = self.config.seq_len
        V = (
            self.config.model_config.vocab_size
            if self.config.model_config
            else self.config.vocab_size
        )
        src = torch.randint(1, V, (B, S), device=self.config.device)
        tgt_in = torch.randint(1, V, (B, S), device=self.config.device)
        tgt_out = torch.randint(1, V, (B, S), device=self.config.device)
        return src, tgt_in, tgt_out

    def _compute_loss_and_grad_norm(
        self, src: torch.Tensor, tgt_in: torch.Tensor, tgt_out: torch.Tensor
    ) -> tuple:
        self.optimizer.zero_grad()
        output = self.model(src, tgt_in)
        logits = output.logits
        B, T, V = logits.shape
        loss = self.criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        self.optimizer.step()
        return loss.item(), grad_norm

    def train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        src, tgt_in, tgt_out = self._make_dummy_batch()
        loss, grad_norm = self._compute_loss_and_grad_norm(src, tgt_in, tgt_out)
        self.scheduler.step()
        return loss, grad_norm

    def _train_epoch_with_loader(self, data_loader, epoch: int, max_steps: int = None) -> tuple:
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        steps = 0

        for batch in data_loader:
            if self.cancel_event.is_set():
                break
            if max_steps is not None and steps >= max_steps:
                break

            src, tgt_in, tgt_out = batch
            src = src.to(self.config.device)
            tgt_in = tgt_in.to(self.config.device)
            tgt_out = tgt_out.to(self.config.device)

            loss, grad_norm = self._compute_loss_and_grad_norm(src, tgt_in, tgt_out)
            total_loss += loss
            total_grad_norm += grad_norm
            steps += 1

        self.scheduler.step()
        n = max(steps, 1)
        return total_loss / n, total_grad_norm / n

    def train(
        self,
        data_loader=None,
        max_steps: int = None,
        save_dir: str = None,
        resume_from: str = None,
        log_file: str = None,
    ) -> list:
        from app.services.train.checkpoint_manager import CheckpointManager
        from app.services.train.training_logger import TrainingLogger

        ckpt_mgr = CheckpointManager()
        logger = TrainingLogger(log_file=log_file)

        start_epoch = 0
        if resume_from is not None:
            start_epoch = ckpt_mgr.load(self, resume_from)
            print(f"Resuming from epoch {start_epoch}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        history = []
        for epoch in range(start_epoch + 1, self.config.num_epochs + 1):
            if self.cancel_event.is_set():
                break

            if data_loader is not None:
                loss, grad_norm = self._train_epoch_with_loader(
                    data_loader, epoch, max_steps=max_steps
                )
            else:
                loss, grad_norm = self.train_epoch(epoch)

            metrics = {
                "epoch": epoch,
                "loss": round(loss, 6),
                "lr": self.scheduler.get_last_lr()[0],
                "grad_norm": round(grad_norm, 6),
                "stage": self.stage,
                "progress": round(epoch / self.config.num_epochs * 100, 1),
            }
            history.append(metrics)
            logger.log(metrics)

            if save_dir is not None:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
                ckpt_mgr.save(self, ckpt_path, epoch=epoch, loss=loss)
                ckpt_mgr.save_best(self, save_dir, metric=loss, epoch=epoch)

            if self.metrics_callback:
                self.metrics_callback(metrics)

        return history
