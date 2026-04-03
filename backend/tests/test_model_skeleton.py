import threading

import pytest
import torch

from app.model.branches import ITransformer, CTransformer, OTransformer
from app.model.tri_transformer import TriTransformerConfig, TriTransformerModel, TriTransformerOutput
from app.model.trainer import TrainerConfig, TriTransformerTrainer, STAGE_MAP


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
def model(small_config):
    return TriTransformerModel(small_config)


@pytest.fixture
def trainer_config(small_config):
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


class TestITransformer:
    def test_output_shape(self):
        branch = ITransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        out, _ = branch(src)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_with_padding_mask(self):
        branch = ITransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[:, -2:] = True
        out, _ = branch(src, src_key_padding_mask=mask)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_with_control_signal(self):
        branch = ITransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        ctrl = torch.randn(BATCH, D_MODEL)
        out, _ = branch(src, control_signal=ctrl)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_no_nan_in_output(self):
        branch = ITransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        out, _ = branch(src)
        assert not torch.isnan(out).any()


class TestCTransformer:
    def test_output_ctrl_signal_shape(self):
        branch = CTransformer(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        i_enc = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl, adaln_i, adaln_o = branch(i_enc)
        assert ctrl.shape == (BATCH, D_MODEL)

    def test_with_o_prev(self):
        branch = CTransformer(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        i_enc = torch.randn(BATCH, SEQ, D_MODEL)
        o_prev = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl, _, _ = branch(i_enc, o_prev=o_prev)
        assert ctrl.shape == (BATCH, D_MODEL)

    def test_without_o_prev(self):
        branch = CTransformer(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        i_enc = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl, _, _ = branch(i_enc, o_prev=None)
        assert ctrl.shape == (BATCH, D_MODEL)

    def test_no_nan(self):
        branch = CTransformer(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        i_enc = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl, _, _ = branch(i_enc)
        assert not torch.isnan(ctrl).any()

    def test_adaln_keys(self):
        branch = CTransformer(d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_layers=2, intermediate_size=128, dropout=0.0)
        i_enc = torch.randn(BATCH, SEQ, D_MODEL)
        _, adaln_i, adaln_o = branch(i_enc)
        for key in ("shift_a", "scale_a", "gate_a", "shift_m", "scale_m", "gate_m"):
            assert key in adaln_i
            assert key in adaln_o


class TestOTransformer:
    def test_output_shapes(self):
        branch = OTransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_plan_layers=2, num_dec_layers=2, intermediate_size=128, dropout=0.0)
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        memory = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl = torch.randn(BATCH, D_MODEL)
        logits, hidden, _ = branch(tgt, memory, control_signal=ctrl)
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert hidden.shape == (BATCH, SEQ, D_MODEL)

    def test_no_nan(self):
        branch = OTransformer(VOCAB, d_model=D_MODEL, num_heads=NUM_HEADS, num_kv_heads=4,
                              num_plan_layers=2, num_dec_layers=2, intermediate_size=128, dropout=0.0)
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        memory = torch.randn(BATCH, SEQ, D_MODEL)
        ctrl = torch.randn(BATCH, D_MODEL)
        logits, _, _ = branch(tgt, memory, control_signal=ctrl)
        assert not torch.isnan(logits).any()


class TestTriTransformerModel:
    def test_forward_output_types(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert isinstance(out, TriTransformerOutput)

    def test_logits_shape(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)

    def test_hidden_shapes(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert out.i_hidden.shape == (BATCH, SEQ, D_MODEL)
        assert out.ctrl_signal.shape == (BATCH, D_MODEL)
        assert out.o_hidden.shape == (BATCH, SEQ, D_MODEL)

    def test_no_nan(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert not torch.isnan(out.logits).any()

    def test_num_parameters(self, model):
        total = model.num_parameters()
        trainable = model.num_parameters(trainable_only=True)
        assert total > 0
        assert trainable <= total

    def test_padding_idx_zero(self, model):
        assert model.i_branch.embedding.padding_idx == 0
        assert model.o_branch.embedding.padding_idx == 0

    def test_with_o_prev_feedback(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        o_prev = torch.randn(BATCH, SEQ, D_MODEL)
        out = model(src, tgt, o_prev=o_prev)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)

    def test_backward_pass(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        labels = torch.randint(0, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        B, T, V = out.logits.shape
        loss = torch.nn.functional.cross_entropy(out.logits.reshape(B * T, V), labels.reshape(B * T))
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestTriTransformerConfig:
    def test_defaults(self):
        cfg = TriTransformerConfig()
        assert cfg.vocab_size == 151936
        assert cfg.d_model == 512
        assert cfg.pad_token_id == 0

    def test_custom(self):
        cfg = TriTransformerConfig(vocab_size=VOCAB, d_model=D_MODEL)
        assert cfg.vocab_size == VOCAB
        assert cfg.d_model == D_MODEL


class TestStageMap:
    def test_known_keys(self):
        assert STAGE_MAP["lora_finetune"] == "stage1"
        assert STAGE_MAP["full_finetune"] == "stage2"
        assert STAGE_MAP["rag_adapt"] == "stage3"
        assert STAGE_MAP["dpo_align"] == "stage3"

    def test_direct_stage_keys(self):
        assert STAGE_MAP["stage1"] == "stage1"
        assert STAGE_MAP["stage2"] == "stage2"
        assert STAGE_MAP["stage3"] == "stage3"


class TestTriTransformerTrainer:
    def test_stage1_freezes_c_branch(self, trainer_config):
        trainer_config.job_type = "lora_finetune"
        trainer = TriTransformerTrainer(trainer_config)
        for p in trainer.model.c_branch.parameters():
            assert not p.requires_grad
        for p in trainer.model.i_branch.parameters():
            assert p.requires_grad

    def test_stage2_freezes_i_and_o_branches(self, trainer_config):
        trainer_config.job_type = "full_finetune"
        trainer = TriTransformerTrainer(trainer_config)
        for p in trainer.model.i_branch.parameters():
            assert not p.requires_grad
        for p in trainer.model.o_branch.parameters():
            assert not p.requires_grad
        for p in trainer.model.c_branch.parameters():
            assert p.requires_grad

    def test_stage3_all_trainable(self, trainer_config):
        trainer_config.job_type = "rag_adapt"
        trainer = TriTransformerTrainer(trainer_config)
        for p in trainer.model.parameters():
            assert p.requires_grad

    def test_train_epoch_returns_float(self, trainer_config):
        trainer = TriTransformerTrainer(trainer_config)
        result = trainer.train_epoch(1)
        loss, grad_norm = result
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0

    def test_train_returns_history(self, trainer_config):
        trainer_config.num_epochs = 2
        trainer = TriTransformerTrainer(trainer_config)
        history = trainer.train()
        assert len(history) == 2
        for m in history:
            assert "epoch" in m
            assert "loss" in m
            assert "lr" in m
            assert "stage" in m
            assert "progress" in m

    def test_cancel_event_stops_training(self, trainer_config):
        trainer_config.num_epochs = 5
        cancel = threading.Event()
        trainer = TriTransformerTrainer(trainer_config, cancel_event=cancel)
        cancel.set()
        history = trainer.train()
        assert len(history) == 0

    def test_metrics_callback_called(self, trainer_config):
        calls = []
        trainer_config.num_epochs = 2
        trainer = TriTransformerTrainer(trainer_config, metrics_callback=calls.append)
        trainer.train()
        assert len(calls) == 2

    def test_progress_values(self, trainer_config):
        trainer_config.num_epochs = 2
        trainer = TriTransformerTrainer(trainer_config)
        history = trainer.train()
        assert history[0]["progress"] == 50.0
        assert history[1]["progress"] == 100.0

    def test_dummy_batch_shapes(self, trainer_config):
        trainer = TriTransformerTrainer(trainer_config)
        src, tgt_in, tgt_out = trainer._make_dummy_batch()
        assert src.shape == (BATCH, SEQ)
        assert tgt_in.shape == (BATCH, SEQ)
        assert tgt_out.shape == (BATCH, SEQ)
