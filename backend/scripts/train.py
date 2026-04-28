#!/usr/bin/env python3
"""
Tri-Transformer 命令行训练入口

用法:
    python backend/scripts/train.py --dataset lccc --epochs 3
    python backend/scripts/train.py --dataset belle --epochs 1 --max-steps 50
    python backend/scripts/train.py --dataset dummy --epochs 2 --batch-size 4
    python backend/scripts/train.py --jetson-nano --dataset dummy
    python backend/scripts/train.py --config backend/configs/jetson_nano_config.yaml
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.model.tokenizer.text_tokenizer import TextTokenizer  # noqa: E402
from app.model.trainer import TriTransformerTrainer, TrainerConfig  # noqa: E402
from app.model.tri_transformer import TriTransformerConfig  # noqa: E402
from app.services.train.dataset_loader import ModelScopeDatasetLoader  # noqa: E402


def _load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _detect_jetson_nano() -> bool:
    try:
        from app.model.jetson_device import detect_jetson_device
        info = detect_jetson_device()
        if info.is_jetson:
            print(f"  检测到 Jetson 设备: {info.gpu_name} (CUDA {info.cuda_version}, {info.total_memory_gb}GB)")
            return True
    except Exception:
        pass
    return False


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
    parser.add_argument("--save-dir", default=None, help="Checkpoint save directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--log-file", default=None, help="Training log JSONL file path")
    parser.add_argument("--jetson-nano", action="store_true",
                        help="Enable Jetson Nano mode (auto-detect if flag set)")
    parser.add_argument("--galore-rank", type=int, default=None,
                        help="GaLore projection rank (Jetson default: 64, standard default: 128)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (Jetson default: 4)")
    parser.add_argument("--config", default=None,
                        help="YAML config file path (e.g. backend/configs/jetson_nano_config.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()

    jetson_nano = args.jetson_nano
    yaml_config = {}
    if args.config:
        yaml_config = _load_yaml_config(args.config)
        if yaml_config.get("jetson_nano", False):
            jetson_nano = True

    if jetson_nano and not args.jetson_nano:
        detected = _detect_jetson_nano()
        if detected:
            jetson_nano = True
            print("  自动启用 Jetson Nano 训练模式")

    if args.jetson_nano:
        detected = _detect_jetson_nano()
        if not detected:
            print("  ⚠️  --jetson-nano 已指定但未检测到 Jetson 设备，仍使用 Jetson 配置")

    effective_batch = yaml_config.get("training", {}).get("batch_size", args.batch_size)
    effective_grad_accum = yaml_config.get("training", {}).get("gradient_accumulation_steps", args.grad_accum)
    effective_device = yaml_config.get("training", {}).get("device", args.device)
    effective_lr = yaml_config.get("training", {}).get("learning_rate", args.lr)
    effective_epochs = yaml_config.get("training", {}).get("num_epochs", args.epochs)
    effective_seq_len = yaml_config.get("training", {}).get("seq_len", args.max_len)
    effective_job_type = yaml_config.get("training", {}).get("job_type", args.job_type)
    effective_d_model = yaml_config.get("model", {}).get("d_model", args.d_model)

    galore_rank = args.galore_rank
    if galore_rank is None:
        galore_rank = yaml_config.get("galore", {}).get("rank", 64 if jetson_nano else 128)

    if jetson_nano:
        effective_batch = min(effective_batch, 1) if effective_batch > 1 else effective_batch
        effective_grad_accum = max(effective_grad_accum, 4)
        effective_device = "cuda"

    print("=== Tri-Transformer Training ===")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Epochs    : {effective_epochs}")
    print(f"  Batch size: {effective_batch}")
    print(f"  Grad accum: {effective_grad_accum}")
    print(f"  Device    : {effective_device}")
    print(f"  Jetson    : {jetson_nano}")
    print(f"  GaLore    : rank={galore_rank}")
    print(f"  Max steps : {args.max_steps}")
    print()

    print("📦 初始化 tokenizer...")
    tokenizer = TextTokenizer(offline=False)
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size: {vocab_size}")

    model_cfg = TriTransformerConfig(
        vocab_size=yaml_config.get("model", {}).get("vocab_size", vocab_size),
        d_model=effective_d_model,
        num_heads=yaml_config.get("model", {}).get("num_heads", 8),
        num_kv_heads=yaml_config.get("model", {}).get("num_kv_heads", 2),
        num_layers_i=yaml_config.get("model", {}).get("num_layers_i", 4),
        num_layers_c=yaml_config.get("model", {}).get("num_layers_c", 4),
        num_plan_layers_o=2,
        num_dec_layers_o=yaml_config.get("model", {}).get("num_layers_o", 4),
        intermediate_size=effective_d_model * 3,
        dropout=0.1,
    )

    trainer_cfg = TrainerConfig(
        job_type=effective_job_type,
        num_epochs=effective_epochs,
        learning_rate=effective_lr,
        batch_size=effective_batch,
        seq_len=effective_seq_len,
        vocab_size=model_cfg.vocab_size,
        device=effective_device,
        model_config=model_cfg,
        gradient_accumulation_steps=effective_grad_accum,
        use_amp=jetson_nano or yaml_config.get("training", {}).get("use_amp", True),
        jetson_nano=jetson_nano,
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
            batch_size=effective_batch,
            max_len=effective_seq_len,
        )
        print(f"  DataLoader batches: {len(data_loader)}")
    else:
        print("  使用虚拟数据（dummy 模式）")

    print()
    print("🚀 开始训练...")
    history = trainer.train(
        data_loader=data_loader,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        resume_from=args.resume,
        log_file=args.log_file,
    )

    print()
    print("=== 训练完成 ===")
    for m in history:
        print(f"  Epoch {m['epoch']:3d} | Loss: {m['loss']:.6f} | LR: {m['lr']:.2e} | Stage: {m['stage']}")


if __name__ == "__main__":
    main()
