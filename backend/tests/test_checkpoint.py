import os
import tempfile

import pytest
import torch

from app.model.trainer import TriTransformerTrainer, TrainerConfig
from app.model.tri_transformer import TriTransformerConfig
from app.services.train.checkpoint_manager import CheckpointManager


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
        num_epochs=1,
        learning_rate=1e-4,
        batch_size=2,
        seq_len=8,
        vocab_size=256,
        device="cpu",
        model_config=cfg,
    )
    return TriTransformerTrainer(config=t_cfg)


class TestCheckpointManagerSaveLoad:
    def test_save_load_epoch_roundtrip(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        path = str(tmp_path / "epoch_001.pt")
        ckpt_mgr.save(tiny_trainer, path, epoch=1, loss=0.5)
        epoch = ckpt_mgr.load(tiny_trainer, path)
        assert epoch == 1

    def test_load_restores_model_state_keys(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        path = str(tmp_path / "epoch_001.pt")
        original_keys = set(tiny_trainer.model.state_dict().keys())
        ckpt_mgr.save(tiny_trainer, path, epoch=1, loss=0.5)
        epoch = ckpt_mgr.load(tiny_trainer, path)
        restored_keys = set(tiny_trainer.model.state_dict().keys())
        assert original_keys == restored_keys
        assert epoch == 1

    def test_load_restores_optimizer_state(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        path = str(tmp_path / "epoch_001.pt")
        ckpt_mgr.save(tiny_trainer, path, epoch=2, loss=0.3)
        epoch = ckpt_mgr.load(tiny_trainer, path)
        assert tiny_trainer.optimizer.state_dict() is not None
        assert epoch == 2

    def test_load_nonexistent_raises_file_not_found(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        with pytest.raises(FileNotFoundError):
            ckpt_mgr.load(tiny_trainer, str(tmp_path / "nonexistent.pt"))


class TestCheckpointManagerSaveBestListCheckpoints:
    def test_save_best_updates_on_improvement(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        save_dir = str(tmp_path)
        updated = ckpt_mgr.save_best(tiny_trainer, save_dir, metric=1.0, epoch=1)
        assert updated is True
        best_path = os.path.join(save_dir, "best_model.pt")
        assert os.path.exists(best_path)

    def test_save_best_no_update_on_worse_metric(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        save_dir = str(tmp_path)
        ckpt_mgr.save_best(tiny_trainer, save_dir, metric=0.5, epoch=1)
        updated = ckpt_mgr.save_best(tiny_trainer, save_dir, metric=1.0, epoch=2)
        assert updated is False

    def test_list_checkpoints_sorted_by_epoch(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        save_dir = str(tmp_path)
        for epoch in [3, 1, 5, 2]:
            ckpt_mgr.save(tiny_trainer, os.path.join(save_dir, f"epoch_{epoch:03d}.pt"), epoch=epoch, loss=0.5)
        checkpoints = ckpt_mgr.list_checkpoints(save_dir)
        epochs = [int(os.path.basename(p).replace("epoch_", "").replace(".pt", "")) for p in checkpoints]
        assert epochs == sorted(epochs)

    def test_list_checkpoints_excludes_non_pt_files(self, tiny_trainer, tmp_path):
        ckpt_mgr = CheckpointManager()
        save_dir = str(tmp_path)
        ckpt_mgr.save(tiny_trainer, os.path.join(save_dir, "epoch_001.pt"), epoch=1, loss=0.5)
        open(os.path.join(save_dir, "training_log.jsonl"), "w").close()
        open(os.path.join(save_dir, "best_model.pt"), "w").close()
        checkpoints = ckpt_mgr.list_checkpoints(save_dir)
        names = [os.path.basename(p) for p in checkpoints]
        assert "training_log.jsonl" not in names
        assert all(n.startswith("epoch_") for n in names)
