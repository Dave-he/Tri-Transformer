import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.model.tri_transformer import TriTransformerConfig
from app.model.trainer import TrainerConfig, TriTransformerTrainer


VOCAB = 256
D_MODEL = 64
NUM_HEADS = 4
BATCH = 2
SEQ = 8


@pytest.fixture
def small_config():
    return TriTransformerConfig(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_kv_heads=4,
        num_layers_i=2,
        num_layers_c=2,
        num_plan_layers_o=2,
        num_dec_layers_o=2,
        intermediate_size=128,
        dropout=0.0,
        max_len=64,
    )


@pytest.fixture
def trainer_cfg(small_config):
    return TrainerConfig(
        job_type="lora_finetune",
        num_epochs=1,
        learning_rate=1e-3,
        batch_size=BATCH,
        seq_len=SEQ,
        vocab_size=VOCAB,
        device="cpu",
        model_config=small_config,
    )


def _make_synthetic_loader(n_batches=3):
    src = torch.randint(1, VOCAB, (n_batches * BATCH, SEQ))
    tgt_in = torch.randint(1, VOCAB, (n_batches * BATCH, SEQ))
    tgt_out = torch.randint(1, VOCAB, (n_batches * BATCH, SEQ))
    ds = TensorDataset(src, tgt_in, tgt_out)
    return DataLoader(ds, batch_size=BATCH)


class TestTrainerWithDataLoader:
    def test_train_with_dataloader_completes(self, trainer_cfg):
        trainer = TriTransformerTrainer(config=trainer_cfg)
        loader = _make_synthetic_loader()
        history = trainer.train(data_loader=loader)
        assert len(history) == 1

    def test_train_with_dataloader_loss_finite(self, trainer_cfg):
        trainer = TriTransformerTrainer(config=trainer_cfg)
        loader = _make_synthetic_loader()
        history = trainer.train(data_loader=loader)
        loss = history[0]["loss"]
        assert math.isfinite(loss)
        assert loss > 0

    def test_train_without_dataloader_fallback(self, trainer_cfg):
        trainer = TriTransformerTrainer(config=trainer_cfg)
        history = trainer.train(data_loader=None)
        assert len(history) == 1
        assert math.isfinite(history[0]["loss"])

    def test_train_metrics_contain_required_keys(self, trainer_cfg):
        trainer = TriTransformerTrainer(config=trainer_cfg)
        loader = _make_synthetic_loader()
        history = trainer.train(data_loader=loader)
        assert "epoch" in history[0]
        assert "loss" in history[0]
        assert "lr" in history[0]

    def test_train_max_steps_early_stop(self, trainer_cfg):
        trainer = TriTransformerTrainer(config=trainer_cfg)
        large_loader = _make_synthetic_loader(n_batches=100)
        history = trainer.train(data_loader=large_loader, max_steps=2)
        assert len(history) == 1
