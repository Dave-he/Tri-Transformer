#!/usr/bin/env python3
"""
Tri-Transformer 命令行训练入口

用法:
    python backend/scripts/train.py --dataset lccc --epochs 3
    python backend/scripts/train.py --dataset belle --epochs 1 --max-steps 50
    python backend/scripts/train.py --dataset dummy --epochs 2 --batch-size 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.model.tokenizer.text_tokenizer import TextTokenizer
from app.model.trainer import TriTransformerTrainer, TrainerConfig
from app.model.tri_transformer import TriTransformerConfig
from app.services.train.dataset_loader import ModelScopeDatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Tri-Transformer Training")
    parser.add_argument("--dataset", default="dummy", choices=["lccc", "belle", "dummy"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--job-type", default="lora_finetune",
                        choices=["lora_finetune", "full_finetune", "rag_adapt", "dpo_align"])
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Tri-Transformer Training ===")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device    : {args.device}")
    print(f"  Max steps : {args.max_steps}")
    print()

    print("📦 初始化 tokenizer...")
    tokenizer = TextTokenizer(offline=False)
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size: {vocab_size}")

    model_cfg = TriTransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=8,
        num_kv_heads=2,
        num_layers_i=4,
        num_layers_c=4,
        num_plan_layers_o=2,
        num_dec_layers_o=4,
        intermediate_size=args.d_model * 3,
        dropout=0.1,
    )

    trainer_cfg = TrainerConfig(
        job_type=args.job_type,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seq_len=args.max_len,
        vocab_size=vocab_size,
        device=args.device,
        model_config=model_cfg,
    )

    print("🤖 初始化模型...")
    trainer = TriTransformerTrainer(config=trainer_cfg)
    total_params = trainer.model.num_parameters()
    print(f"  总参数量: {total_params:,}")

    data_loader = None
    if args.dataset != "dummy":
        print(f"📂 加载 {args.dataset} 数据集...")
        ds_loader = ModelScopeDatasetLoader()
        dataset = ds_loader.load(args.dataset, max_samples=args.max_samples)
        print(f"  样本数: {len(dataset)}")
        data_loader = ds_loader.get_dataloader(
            dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
        )
        print(f"  DataLoader batches: {len(data_loader)}")
    else:
        print("  使用虚拟数据（dummy 模式）")

    print()
    print("🚀 开始训练...")
    history = trainer.train(data_loader=data_loader, max_steps=args.max_steps)

    print()
    print("=== 训练完成 ===")
    for m in history:
        print(f"  Epoch {m['epoch']:3d} | Loss: {m['loss']:.6f} | LR: {m['lr']:.2e} | Stage: {m['stage']}")


if __name__ == "__main__":
    main()
