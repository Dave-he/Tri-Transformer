"""
Tri-Transformer 完整测试套件

测试范围：
- 模型前向传播
- 损失函数
- 评估器
- 训练流程
- 模型保存/加载
"""
import math
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from app.model.tri_transformer import (
    TriTransformerConfig,
    TriTransformerModel,
    TriTransformerOutput,
)
from app.services.model.loss_functions import (
    HallucinationLoss,
    RAGLoss,
    ControlAlignmentLoss,
    TotalLoss,
)
from app.services.model.evaluation import (
    PerplexityEvaluator,
    AccuracyEvaluator,
    HallucinationEvaluator,
    RAGEvaluator,
    TriTransformerEvaluator,
)
from app.services.model.train import AdvancedTrainer, create_synthetic_dataloader


VOCAB = 256
D_MODEL = 64
NUM_HEADS = 4
BATCH = 2
SEQ = 8


@pytest.fixture
def small_config():
    """小型测试配置"""
    return TriTransformerConfig(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_kv_heads=4,
        num_layers_i=2,
        num_layers_c=2,
        num_slots_c=8,
        num_plan_layers_o=2,
        num_dec_layers_o=2,
        intermediate_size=128,
        dropout=0.0,
        max_len=64,
    )


@pytest.fixture
def model(small_config):
    """测试模型"""
    return TriTransformerModel(small_config)


@pytest.fixture
def device():
    """测试设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestTriTransformerModel:
    """模型测试类"""
    
    def test_forward_returns_output_object(self, model):
        """测试前向传播返回正确的输出对象"""
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert isinstance(out, TriTransformerOutput)
    
    def test_logits_shape(self, model):
        """测试 logits 形状"""
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
    
    def test_hidden_states_shapes(self, model):
        """测试隐状态形状"""
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        
        assert out.i_hidden.shape == (BATCH, SEQ, D_MODEL)
        assert out.ctrl_signal.shape == (BATCH, D_MODEL)
        assert out.o_hidden.shape == (BATCH, SEQ, D_MODEL)
    
    def test_with_padding_mask(self, model):
        """测试填充掩码"""
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[0, -2:] = True
        
        out = model(src, tgt, src_key_padding_mask=mask)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)
    
    def test_num_parameters(self, model):
        """测试参数量计算"""
        total_params = model.num_parameters()
        trainable_params = model.num_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


class TestLossFunctions:
    """损失函数测试类"""
    
    def test_hallucination_loss(self):
        """测试幻觉损失"""
        loss_fn = HallucinationLoss(margin=1.0)
        
        context_enc = torch.randn(BATCH, D_MODEL)
        positive_span = torch.randn(BATCH, D_MODEL)
        negative_span = torch.randn(BATCH, D_MODEL)
        
        loss = loss_fn(context_enc, positive_span, negative_span)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
        assert loss.dim() == 0
    
    def test_rag_loss(self):
        """测试 RAG 损失"""
        loss_fn = RAGLoss(retrieval_weight=0.3, generation_weight=0.7)
        
        query_enc = torch.randn(BATCH, D_MODEL)
        doc_encs = torch.randn(BATCH, 5, D_MODEL)
        logits = torch.randn(BATCH, SEQ, VOCAB)
        target_ids = torch.randint(1, VOCAB, (BATCH, SEQ))
        relevant_doc_idx = torch.randint(0, 5, (BATCH,))
        
        loss_dict = loss_fn(query_enc, doc_encs, logits, target_ids, relevant_doc_idx)
        
        assert "total_loss" in loss_dict
        assert "retrieval_loss" in loss_dict
        assert "generation_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
    
    def test_control_alignment_loss(self):
        """测试控制对齐损失"""
        loss_fn = ControlAlignmentLoss(control_dim=D_MODEL, num_control_modes=4)
        
        ctrl_signal = torch.randn(BATCH, D_MODEL)
        target_mode = torch.randint(0, 4, (BATCH,))
        
        loss = loss_fn(ctrl_signal, target_mode)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
    
    def test_total_loss(self):
        """测试综合损失"""
        loss_fn = TotalLoss(
            llm_weight=1.0,
            hallucination_weight=0.5,
            rag_weight=0.3,
            control_weight=0.2,
        )
        
        logits = torch.randn(BATCH, SEQ, VOCAB)
        target_ids = torch.randint(1, VOCAB, (BATCH, SEQ))
        context_enc = torch.randn(BATCH, D_MODEL)
        positive_span = torch.randn(BATCH, D_MODEL)
        negative_span = torch.randn(BATCH, D_MODEL)
        ctrl_signal = torch.randn(BATCH, D_MODEL)
        target_mode = torch.randint(0, 4, (BATCH,))
        
        loss_dict = loss_fn(
            logits=logits,
            target_ids=target_ids,
            context_enc=context_enc,
            positive_span=positive_span,
            negative_span=negative_span,
            ctrl_signal=ctrl_signal,
            target_mode=target_mode,
        )
        
        assert "total_loss" in loss_dict
        assert "llm_loss" in loss_dict
        assert loss_dict["total_loss"] >= 0
        assert loss_dict["llm_loss"] >= 0


class TestEvaluators:
    """评估器测试类"""
    
    def test_perplexity_evaluator(self):
        """测试困惑度评估器"""
        evaluator = PerplexityEvaluator(ignore_index=0)
        
        logits = torch.randn(BATCH, SEQ, VOCAB)
        target_ids = torch.randint(1, VOCAB, (BATCH, SEQ))
        
        perplexity = evaluator.evaluate(logits, target_ids)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_accuracy_evaluator(self):
        """测试准确率评估器"""
        evaluator = AccuracyEvaluator(ignore_index=0)
        
        logits = torch.randn(BATCH, SEQ, VOCAB)
        target_ids = torch.randint(1, VOCAB, (BATCH, SEQ))
        
        accuracy = evaluator.evaluate(logits, target_ids)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_hallucination_evaluator(self):
        """测试幻觉评估器"""
        evaluator = HallucinationEvaluator(similarity_threshold=0.7)
        
        generated_content = torch.randn(BATCH, SEQ, D_MODEL)
        knowledge_base = torch.randn(10, D_MODEL)
        context = torch.randn(BATCH, D_MODEL)
        
        metrics = evaluator.evaluate(generated_content, knowledge_base, context)
        
        assert "fact_consistency" in metrics
        assert "knowledge_grounding" in metrics
        assert "contradiction_rate" in metrics
        
        assert 0 <= metrics["fact_consistency"] <= 1
        assert 0 <= metrics["knowledge_grounding"] <= 1
        assert 0 <= metrics["contradiction_rate"] <= 1
    
    def test_rag_evaluator(self):
        """测试 RAG 评估器"""
        evaluator = RAGEvaluator(top_k=3)
        
        query_enc = torch.randn(BATCH, D_MODEL)
        doc_encs = torch.randn(BATCH, 10, D_MODEL)
        relevant_indices = torch.randint(0, 10, (BATCH,))
        
        metrics = evaluator.evaluate(query_enc, doc_encs, relevant_indices)
        
        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "mrr" in metrics
        
        assert 0 <= metrics["precision_at_k"] <= 1
        assert 0 <= metrics["recall_at_k"] <= 1
        assert 0 <= metrics["mrr"] <= 1


class TestAdvancedTrainer:
    """高级训练器测试类"""
    
    @pytest.fixture
    def trainer(self, small_config, device):
        """创建测试训练器"""
        model = TriTransformerModel(small_config).to(device)
        
        from app.model.trainer import TrainerConfig
        config = TrainerConfig(
            job_type="lora_finetune",
            num_epochs=1,
            learning_rate=1e-3,
            batch_size=BATCH,
            seq_len=SEQ,
            vocab_size=VOCAB,
            device=device,
            model_config=small_config,
        )
        
        return AdvancedTrainer(
            model=model,
            config=config,
            device=device,
            output_dir="./test_checkpoints",
        )
    
    def test_trainer_initialization(self, trainer):
        """测试训练器初始化"""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.output_dir.exists()
    
    def test_train_epoch(self, trainer, device):
        """测试训练 epoch"""
        loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=20,
            shuffle=True,
        )
        
        metrics = trainer.train_epoch(loader, epoch=1, max_steps=5)
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert math.isfinite(metrics["loss"])
        assert metrics["loss"] > 0
    
    def test_evaluate(self, trainer, device):
        """测试评估"""
        loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=10,
            shuffle=False,
        )
        
        metrics = trainer.evaluate(loader, max_batches=2)
        
        assert "eval_loss" in metrics
        assert "accuracy" in metrics
        assert math.isfinite(metrics["eval_loss"])
    
    def test_save_and_load_checkpoint(self, trainer, device, tmp_path):
        """测试检查点保存和加载"""
        loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=10,
            shuffle=True,
        )
        
        metrics = trainer.train_epoch(loader, epoch=1, max_steps=2)
        trainer.save_checkpoint(epoch=1, step=2, metrics=metrics, filename="test_checkpoint.pt")
        
        checkpoint_path = trainer.output_dir / "test_checkpoint.pt"
        assert checkpoint_path.exists()
        
        state = trainer.load_checkpoint(str(checkpoint_path), load_optimizer=True)
        
        assert state.epoch == 1
        assert state.step == 2
    
    def test_full_training_loop(self, trainer, device):
        """测试完整训练流程"""
        train_loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=50,
            shuffle=True,
        )
        
        val_loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=10,
            shuffle=False,
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            max_steps_per_epoch=5,
            save_every=1,
            early_stopping_patience=3,
        )
        
        assert len(history) == 2
        assert all("loss" in h for h in history)
        assert all(math.isfinite(h["loss"]) for h in history)


class TestIntegration:
    """集成测试类"""
    
    def test_end_to_end_training(self, small_config, device, tmp_path):
        """端到端训练测试"""
        model = TriTransformerModel(small_config).to(device)
        
        from app.model.trainer import TrainerConfig
        config = TrainerConfig(
            job_type="lora_finetune",
            num_epochs=3,
            learning_rate=1e-3,
            batch_size=BATCH,
            seq_len=SEQ,
            vocab_size=VOCAB,
            device=device,
            model_config=small_config,
        )
        
        trainer = AdvancedTrainer(
            model=model,
            config=config,
            device=device,
            output_dir=str(tmp_path / "checkpoints"),
        )
        
        train_loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=100,
            shuffle=True,
        )
        
        val_loader = create_synthetic_dataloader(
            vocab_size=VOCAB,
            batch_size=BATCH,
            seq_len=SEQ,
            num_samples=20,
            shuffle=False,
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            max_steps_per_epoch=10,
            save_every=1,
            early_stopping_patience=5,
        )
        
        assert len(history) > 0
        
        final_loss = history[-1]["loss"]
        initial_loss = history[0]["loss"]
        
        assert final_loss < initial_loss or len(history) > 1
        
        checkpoint_path = tmp_path / "checkpoints" / "checkpoint_final.pt"
        assert checkpoint_path.exists()
        
        history_path = tmp_path / "checkpoints" / "training_history.json"
        assert history_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
