"""
Tri-Transformer 快速训练启动脚本

使用方法：
    # 轻量级测试（CPU/GPU）
    python -m app.services.model.quick_start --config lightweight --epochs 5
    
    # GPU 训练（推荐）
    python -m app.services.model.quick_start --config lightweight --epochs 10 --batch-size 16 --use-amp
    
    # 恢复训练
    python -m app.services.model.quick_start --resume ./checkpoints/checkpoint_latest.pt
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "backend"))

import torch
import json
from datetime import datetime

from app.model.tri_transformer import (
    TriTransformerConfig,
    TriTransformerModel,
    QWEN3_8B_CONFIG,
    QWEN3_30B_CONFIG,
)
from app.model.trainer import TrainerConfig
from app.services.model.train import AdvancedTrainer, create_synthetic_dataloader


def print_training_summary(history, model, config):
    """打印训练摘要"""
    print("\n" + "="*70)
    print("📊 训练摘要".center(70))
    print("="*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数量:")
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数量：{trainable_params:,}")
    print(f"  参数量 (MB): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\n训练配置:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Sequence Length: {config.seq_len}")
    
    if history:
        print(f"\n训练结果:")
        print(f"  初始 Loss: {history[0]['loss']:.6f}")
        print(f"  最终 Loss: {history[-1]['loss']:.6f}")
        print(f"  最终 Accuracy: {history[-1]['accuracy']:.4f}")
        
        if len(history) > 1:
            loss_improvement = (history[0]['loss'] - history[-1]['loss']) / history[0]['loss'] * 100
            print(f"  Loss 改善：{loss_improvement:.2f}%")
        
        if "perplexity" in history[-1]:
            print(f"  最终 Perplexity: {history[-1]['perplexity']:.2f}")
    
    print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tri-Transformer 快速训练启动")
    parser.add_argument(
        "--config",
        type=str,
        default="lightweight",
        choices=["lightweight", "qwen3-8b", "qwen3-30b"],
        help="模型配置",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--seq-len", type=int, default=64, help="序列长度")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    parser.add_argument("--use-amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--num-samples", type=int, default=1000, help="训练样本数")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--max-steps", type=int, default=None, help="每 epoch 最大步数")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    
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
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 使用设备：{device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    print(f"\n📦 加载模型配置：{args.config}")
    model = TriTransformerModel(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量：{total_params:,}")
    print(f"   可训练参数量：{trainable_params:,}")
    
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
        print(f"\n📥 从检查点恢复：{args.resume}")
        state = trainer.load_checkpoint(args.resume)
        print(f"   恢复至 epoch {state.epoch}, step {state.step}")
    
    num_train_samples = int(args.num_samples * (1 - args.val_split))
    num_val_samples = args.num_samples - num_train_samples
    
    print(f"\n📊 数据集:")
    print(f"   训练样本：{num_train_samples:,}")
    print(f"   验证样本：{num_val_samples:,}")
    
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
    
    print(f"\n🏋️  开始训练...\n")
    
    start_time = datetime.now()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        max_steps_per_epoch=args.max_steps,
        save_every=1,
        early_stopping_patience=3,
        progress_callback=lambda m: None,
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print(f"\n⏱️  训练完成!")
    print(f"   总耗时：{training_duration}")
    
    print_training_summary(history, model, trainer_config)
    
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"📁 检查点已保存至：{args.output_dir}")
    print(f"📊 训练历史已保存至：{history_path}")
    
    print("\n✅ 训练完成！")
    print("\n下一步:")
    print("  1. 查看训练日志：tensorboard --logdir ./checkpoints")
    print("  2. 评估模型：python -m app.services.model.evaluate --checkpoint ./checkpoints/checkpoint_best.pt")
    print("  3. 部署模型：python -m app.services.model.inference --checkpoint ./checkpoints/checkpoint_best.pt")


if __name__ == "__main__":
    main()
