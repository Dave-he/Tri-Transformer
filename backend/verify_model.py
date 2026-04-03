"""
Tri-Transformer 模型快速验证脚本
不依赖 pytest，直接验证核心功能
"""
import torch
import sys
from pathlib import Path

# 添加 backend 到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.model.tri_transformer import TriTransformerConfig, TriTransformerModel
from app.services.model.loss_functions import TotalLoss
from app.services.model.evaluation import TriTransformerEvaluator

print("="*70)
print("Tri-Transformer 模型验证".center(70))
print("="*70)

# 创建小型配置
print("\n[1/8] 创建模型配置...")
config = TriTransformerConfig(
    vocab_size=1000,
    d_model=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers_i=2,
    num_layers_c=2,
    num_slots_c=8,
    num_plan_layers_o=2,
    num_dec_layers_o=2,
    intermediate_size=256,
    dropout=0.1,
    max_len=128,
)
print("   [OK] 配置创建成功")
print(f"        - 词表大小：{config.vocab_size:,}")
print(f"        - 隐藏维度：{config.d_model}")
print(f"        - 注意力头数：{config.num_heads}")

# 创建模型
print("\n[2/8] 初始化模型...")
model = TriTransformerModel(config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("   [OK] 模型初始化成功")
print(f"        - 总参数量：{total_params:,}")
print(f"        - 可训练参数：{trainable_params:,}")

# 前向传播测试
print("\n[3/8] 测试前向传播...")
batch_size = 2
seq_len = 16
src = torch.randint(1, config.vocab_size, (batch_size, seq_len))
tgt = torch.randint(1, config.vocab_size, (batch_size, seq_len))

print(f"   输入形状:")
print(f"        - src: {src.shape}")
print(f"        - tgt: {tgt.shape}")

with torch.no_grad():
    output = model(src, tgt)

print(f"   输出形状:")
print(f"        - logits: {output.logits.shape}")
print(f"        - i_hidden: {output.i_hidden.shape}")
print(f"        - ctrl_signal: {output.ctrl_signal.shape}")
print(f"        - o_hidden: {output.o_hidden.shape}")
print("   [OK] 前向传播成功")

# 损失函数测试
print("\n[4/8] 测试损失函数...")
target_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
loss_fn = TotalLoss(
    llm_weight=1.0,
    hallucination_weight=0.5,
    rag_weight=0.3,
    control_weight=0.2,
)

loss_dict = loss_fn(
    logits=output.logits,
    target_ids=target_ids,
    context_enc=output.i_hidden.mean(dim=1),
    ctrl_signal=output.ctrl_signal,
)

print(f"   损失值:")
print(f"        - Total Loss: {loss_dict['total_loss']:.4f}")
print(f"        - LLM Loss: {loss_dict['llm_loss']:.4f}")
if loss_dict.get('hallucination_loss') is not None:
    print(f"        - Hallucination Loss: {loss_dict['hallucination_loss']:.4f}")
if loss_dict.get('control_loss') is not None:
    print(f"        - Control Loss: {loss_dict['control_loss']:.4f}")
print("   [OK] 损失计算成功")

# 评估器测试
print("\n[5/8] 测试评估器...")
evaluator = TriTransformerEvaluator(config)
metrics = evaluator.evaluate(output, target_ids)

print(f"   评估指标:")
print(f"        - Perplexity: {metrics.get('perplexity', 'N/A')}")
print(f"        - Accuracy: {metrics.get('accuracy', 'N/A')}")
print("   [OK] 评估完成")

# 训练测试
print("\n[6/8] 测试训练流程...")
from app.model.trainer import TrainerConfig
from app.services.model.train import AdvancedTrainer, create_synthetic_dataloader

trainer_config = TrainerConfig(
    job_type="lora_finetune",
    num_epochs=1,
    learning_rate=1e-3,
    batch_size=batch_size,
    seq_len=seq_len,
    vocab_size=config.vocab_size,
    device="cpu",
    model_config=config,
)

trainer = AdvancedTrainer(
    model=model,
    config=trainer_config,
    device="cpu",
    output_dir="./test_checkpoints",
)

train_loader = create_synthetic_dataloader(
    vocab_size=config.vocab_size,
    batch_size=batch_size,
    seq_len=seq_len,
    num_samples=20,
    shuffle=True,
)

print(f"   训练配置:")
print(f"        - Epochs: {trainer_config.num_epochs}")
print(f"        - Batch Size: {trainer_config.batch_size}")
print(f"        - Learning Rate: {trainer_config.learning_rate}")

train_metrics = trainer.train_epoch(train_loader, epoch=1, max_steps=5)
print(f"   训练结果:")
print(f"        - Loss: {train_metrics['loss']:.6f}")
print(f"        - Accuracy: {train_metrics['accuracy']:.4f}")
print("   [OK] 训练测试成功")

# 保存检查点
print("\n[7/8] 测试检查点保存...")
trainer.save_checkpoint(epoch=1, step=5, metrics=train_metrics, filename="test_checkpoint.pt")
print("   [OK] 检查点已保存")

# 加载检查点
print("\n[8/8] 测试检查点加载...")
state = trainer.load_checkpoint("./test_checkpoints/test_checkpoint.pt")
print(f"   [OK] 检查点已加载 (epoch={state.epoch}, step={state.step})")

print("\n" + "="*70)
print("所有验证通过！Tri-Transformer 模型工作正常".center(70))
print("="*70)

print("\n下一步:")
print("  1. 运行完整训练：python -m app.services.model.quick_start --config lightweight --epochs 10")
print("  2. 运行评估：python -m app.services.model.evaluate --checkpoint ./checkpoints/checkpoint_best.pt")
print("  3. 交互式推理：python -m app.services.model.inference_cli --checkpoint ./checkpoints/checkpoint_best.pt --mode interactive")
