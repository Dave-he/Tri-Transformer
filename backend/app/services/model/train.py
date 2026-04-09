"""
Tri-Transformer 完整训练脚本

支持：
- 三分阶段训练（Stage 1/2/3）
- 自定义损失函数
- 评估指标跟踪
- 模型保存/加载
- 训练可视化
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from app.model.tri_transformer import (
    TriTransformerModel,
    TriTransformerConfig,
    QWEN3_8B_CONFIG,
    QWEN3_30B_CONFIG,
)
from app.model.trainer import TrainerConfig, TriTransformerTrainer
from app.services.model.loss_functions import TotalLoss
from app.services.model.evaluation import TriTransformerEvaluator


@dataclass
class TrainingState:
    """训练状态快照"""
    epoch: int
    step: int
    best_loss: float
    best_accuracy: float
    config: dict


class AdvancedTrainer:
    """
    高级训练器，支持完整训练流程
    
    特性：
    - 三阶段训练策略
    - 动态学习率调度
    - 梯度累积
    - 混合精度训练
    - 早停机制
    - 检查点保存
    """
    
    def __init__(
        self,
        model: TriTransformerModel,
        config: TrainerConfig,
        device: Optional[str] = None,
        output_dir: str = "./checkpoints",
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.config = config
        self.device = device or config.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 1e-2,
        )
        
        self.total_loss_fn = TotalLoss(
            llm_weight=1.0,
            hallucination_weight=0.5,
            rag_weight=0.3,
            control_weight=0.2,
        )
        
        self.evaluator = TriTransformerEvaluator(model.config)
        
        self.training_history = []
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.early_stop_counter = 0
        
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        metrics: dict,
        filename: Optional[str] = None,
    ):
        """保存训练检查点"""
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        if metrics.get("loss", float("inf")) < self.best_loss:
            self.best_loss = metrics["loss"]
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"★ New best model saved: {best_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
    ) -> TrainingState:
        """加载训练检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("scaler_state") and self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        state = TrainingState(
            epoch=checkpoint["epoch"],
            step=checkpoint["step"],
            best_loss=checkpoint.get("best_loss", float("inf")),
            best_accuracy=checkpoint.get("best_accuracy", 0.0),
            config=checkpoint.get("config", {}),
        )
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        return state
    
    def train_epoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_llm_loss = 0.0
        total_hall_loss = 0.0
        total_rag_loss = 0.0
        total_ctrl_loss = 0.0
        
        num_batches = 0
        correct_tokens = 0
        total_tokens = 0
        
        pbar = tqdm(
            enumerate(data_loader),
            total=min(len(data_loader), max_steps) if max_steps else len(data_loader),
            desc=f"Epoch {epoch}",
        )
        
        for step, batch in pbar:
            if max_steps and step >= max_steps:
                break
            
            src, tgt_in, tgt_out = batch
            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_out = tgt_out.to(self.device)
            
            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(src, tgt_in)
                    loss_dict = self.total_loss_fn(
                        logits=output.logits,
                        target_ids=tgt_out,
                        context_enc=output.i_hidden.mean(dim=1),
                        ctrl_signal=output.ctrl_signal,
                    )
                    loss = loss_dict["total_loss"] / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(src, tgt_in)
                loss_dict = self.total_loss_fn(
                    logits=output.logits,
                    target_ids=tgt_out,
                    context_enc=output.i_hidden.mean(dim=1),
                    ctrl_signal=output.ctrl_signal,
                )
                loss = loss_dict["total_loss"] / self.gradient_accumulation_steps
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss_dict["total_loss"].item()
            total_llm_loss += loss_dict.get("llm_loss", torch.tensor(0.0)).item()
            
            if loss_dict.get("hallucination_loss") is not None:
                total_hall_loss += loss_dict["hallucination_loss"].item()
            if loss_dict.get("rag_loss") is not None:
                total_rag_loss += loss_dict["rag_loss"].item()
            if loss_dict.get("control_loss") is not None:
                total_ctrl_loss += loss_dict["control_loss"].item()
            
            pred_ids = output.logits.argmax(dim=-1)
            correct_tokens += (pred_ids == tgt_out).sum().item()
            total_tokens += tgt_out.numel()
            
            num_batches += 1
            
            if step % 10 == 0:
                metrics = {
                    "epoch": epoch,
                    "step": step,
                    "loss": round(loss_dict["total_loss"].item(), 6),
                    "llm_loss": round(loss_dict.get("llm_loss", torch.tensor(0.0)).item(), 6),
                    "accuracy": round(correct_tokens / max(total_tokens, 1), 6),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                pbar.set_postfix(metrics)
                
                if progress_callback:
                    progress_callback(metrics)
        
        self.scheduler.step()
        
        avg_metrics = {
            "epoch": epoch,
            "loss": round(total_loss / max(num_batches, 1), 6),
            "llm_loss": round(total_llm_loss / max(num_batches, 1), 6),
            "hallucination_loss": round(total_hall_loss / max(num_batches, 1), 6) if total_hall_loss else None,
            "rag_loss": round(total_rag_loss / max(num_batches, 1), 6) if total_rag_loss else None,
            "control_loss": round(total_ctrl_loss / max(num_batches, 1), 6) if total_ctrl_loss else None,
            "accuracy": round(correct_tokens / max(total_tokens, 1), 6),
            "lr": self.scheduler.get_last_lr()[0],
            "num_batches": num_batches,
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> dict:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in tqdm(data_loader, desc="Evaluating", total=max_batches):
            if max_batches and num_batches >= max_batches:
                break
            
            src, tgt_in, tgt_out = batch
            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_out = tgt_out.to(self.device)
            
            output = self.model(src, tgt_in)
            
            loss_dict = self.total_loss_fn(
                logits=output.logits,
                target_ids=tgt_out,
            )
            
            eval_metrics = self.evaluator.evaluate(
                output,
                tgt_out,
            )
            
            total_loss += loss_dict["total_loss"].item()
            total_perplexity += eval_metrics.get("perplexity", 0.0)
            total_accuracy += eval_metrics.get("accuracy", 0.0)
            num_batches += 1
        
        return {
            "eval_loss": round(total_loss / max(num_batches, 1), 6),
            "perplexity": round(total_perplexity / max(num_batches, 1), 2),
            "accuracy": round(total_accuracy / max(num_batches, 1), 6),
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        max_steps_per_epoch: Optional[int] = None,
        save_every: int = 1,
        early_stopping_patience: int = 3,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        """完整训练流程"""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   Mixed precision: {self.use_amp}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(
                train_loader,
                epoch,
                max_steps=max_steps_per_epoch,
                progress_callback=progress_callback,
            )
            
            epoch_time = time.time() - start_time
            
            if val_loader is not None:
                eval_metrics = self.evaluate(val_loader, max_batches=10)
                train_metrics.update(eval_metrics)
                
                if eval_metrics["accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_metrics["accuracy"]
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            
            train_metrics["epoch_time"] = round(epoch_time, 2)
            self.training_history.append(train_metrics)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Loss: {train_metrics['loss']:.6f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            if "perplexity" in train_metrics:
                print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"{'='*60}\n")
            
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, 0, train_metrics)
            
            if early_stopping_patience and self.early_stop_counter >= early_stopping_patience:
                print(f"⚠️  Early stopping triggered at epoch {epoch}")
                break
        
        print(f"✓ Training completed!")
        print(f"  Best loss: {self.best_loss:.6f}")
        print(f"  Best accuracy: {self.best_accuracy:.4f}")
        
        self.save_checkpoint(
            num_epochs,
            0,
            {"final_loss": self.training_history[-1]["loss"]},
            filename="checkpoint_final.pt",
        )

        history_path = Path(self.output_dir) / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        return self.training_history


def create_synthetic_dataloader(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    num_samples: int = 1000,
    shuffle: bool = True,
) -> DataLoader:
    """创建合成数据加载器用于测试"""
    from torch.utils.data import TensorDataset
    
    src = torch.randint(1, vocab_size, (num_samples, seq_len))
    tgt_in = torch.randint(1, vocab_size, (num_samples, seq_len))
    tgt_out = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    dataset = TensorDataset(src, tgt_in, tgt_out)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    """训练主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Tri-Transformer Model")
    parser.add_argument(
        "--config",
        type=str,
        default="lightweight",
        choices=["lightweight", "qwen3-8b", "qwen3-30b"],
        help="Model configuration",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    
    args = parser.parse_args()
    
    config_map = {
        "lightweight": TriTransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_kv_heads=2,
            num_layers_i=4,
            num_layers_c=2,
            num_slots_c=8,
            num_plan_layers_o=2,
            num_dec_layers_o=4,
            intermediate_size=512,
            dropout=0.1,
            rope_theta=1_000_000.0,
            max_len=512,
        ),
        "qwen3-8b": QWEN3_8B_CONFIG,
        "qwen3-30b": QWEN3_30B_CONFIG,
    }
    
    model_config = config_map[args.config]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = TriTransformerModel(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer_config = TrainerConfig(
        job_type="lora_finetune",
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        device=device,
        model_config=model_config,
    )
    
    trainer = AdvancedTrainer(
        model=model,
        config=trainer_config,
        device=device,
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {state.epoch}, step {state.step}")
    
    num_train_samples = int(args.num_samples * (1 - args.val_split))
    num_val_samples = args.num_samples - num_train_samples
    
    train_loader = create_synthetic_dataloader(
        vocab_size=model_config.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_samples=num_train_samples,
        shuffle=True,
    )
    
    val_loader = None
    if num_val_samples > 0:
        val_loader = create_synthetic_dataloader(
            vocab_size=model_config.vocab_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_samples=num_val_samples,
            shuffle=False,
        )
    
    def progress_callback(metrics):
        pass
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        max_steps_per_epoch=None,
        save_every=1,
        early_stopping_patience=3,
        progress_callback=progress_callback,
    )
    
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")


if __name__ == "__main__":
    main()
