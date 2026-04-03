import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.model.trainer import TriTransformerTrainer, TrainerConfig
from app.model.tri_transformer import TriTransformerConfig


@pytest.fixture
def tiny_trainer():
    cfg = TriTransformerConfig(
        vocab_size=256,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers_i=1,
        num_layers_c=1,
        num_plan_layers_o=1,
        num_dec_layers_o=1,
        intermediate_size=128,
        dropout=0.0,
    )
    t_cfg = TrainerConfig(
        job_type="lora_finetune",
        num_epochs=2,
        learning_rate=1e-4,
        batch_size=2,
        seq_len=8,
        vocab_size=256,
        device="cpu",
        model_config=cfg,
    )
    return TriTransformerTrainer(config=t_cfg)


def make_loader(vocab=256, batch=2, seq=8, steps=3):
    src = torch.randint(1, vocab, (steps * batch, seq))
    tgt_in = torch.randint(1, vocab, (steps * batch, seq))
    tgt_out = torch.randint(1, vocab, (steps * batch, seq))
    ds = TensorDataset(src, tgt_in, tgt_out)
    return DataLoader(ds, batch_size=batch)


class TestTrainerGradNorm:
    def test_metrics_contain_grad_norm(self, tiny_trainer):
        history = tiny_trainer.train()
        for m in history:
            assert "grad_norm" in m, f"grad_norm missing in {m}"

    def test_grad_norm_is_finite(self, tiny_trainer):
        history = tiny_trainer.train()
        for m in history:
            assert isinstance(m["grad_norm"], float)
            assert m["grad_norm"] >= 0.0
            assert m["grad_norm"] < float("inf")

    def test_grad_norm_with_dataloader(self, tiny_trainer):
        loader = make_loader()
        history = tiny_trainer.train(data_loader=loader)
        for m in history:
            assert "grad_norm" in m
            assert m["grad_norm"] >= 0.0


class TestTrainerSaveDir:
    def test_train_with_save_dir_creates_checkpoints(self, tiny_trainer, tmp_path):
        save_dir = str(tmp_path)
        tiny_trainer.train(save_dir=save_dir)
        files = os.listdir(save_dir)
        epoch_files = [f for f in files if f.startswith("epoch_") and f.endswith(".pt")]
        assert len(epoch_files) == 2

    def test_train_with_save_dir_creates_best_model(self, tiny_trainer, tmp_path):
        save_dir = str(tmp_path)
        tiny_trainer.train(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "best_model.pt"))

    def test_train_without_save_dir_no_files(self, tiny_trainer, tmp_path):
        tiny_trainer.train(save_dir=None)
        assert not any(f.endswith(".pt") for f in os.listdir(str(tmp_path)))


class TestTrainerResumeFrom:
    def test_resume_continues_from_correct_epoch(self, tmp_path):
        cfg = TriTransformerConfig(
            vocab_size=256, d_model=64, num_heads=4, num_kv_heads=2,
            num_layers_i=1, num_layers_c=1, num_plan_layers_o=1, num_dec_layers_o=1,
            intermediate_size=128, dropout=0.0,
        )
        t_cfg1 = TrainerConfig(num_epochs=2, learning_rate=1e-4, batch_size=2,
                               seq_len=8, vocab_size=256, device="cpu", model_config=cfg)
        trainer1 = TriTransformerTrainer(config=t_cfg1)
        save_dir = str(tmp_path)
        trainer1.train(save_dir=save_dir)

        resume_path = os.path.join(save_dir, "epoch_002.pt")

        t_cfg2 = TrainerConfig(num_epochs=3, learning_rate=1e-4, batch_size=2,
                               seq_len=8, vocab_size=256, device="cpu", model_config=cfg)
        trainer2 = TriTransformerTrainer(config=t_cfg2)
        history = trainer2.train(resume_from=resume_path)

        assert len(history) == 1
        assert history[0]["epoch"] == 3
