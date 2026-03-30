import pytest
import torch

from app.model.tri_transformer import TriTransformerConfig, TriTransformerModel, TriTransformerOutput

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


class TestTriTransformerForward:
    def test_forward_returns_output_object(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert isinstance(out, TriTransformerOutput)

    def test_logits_shape(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)

    def test_hidden_states_are_tensors(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = model(src, tgt)
        assert isinstance(out.i_hidden, torch.Tensor)
        assert isinstance(out.ctrl_signal, torch.Tensor)
        assert isinstance(out.o_hidden, torch.Tensor)

    def test_no_unpacking_error(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        try:
            model(src, tgt)
        except (ValueError, TypeError) as e:
            pytest.fail(f"forward() raised unexpected error: {e}")

    def test_with_padding_mask(self, model):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
        mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        mask[0, -2:] = True
        out = model(src, tgt, src_key_padding_mask=mask)
        assert out.logits.shape == (BATCH, SEQ, VOCAB)

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0
        assert model.num_parameters(trainable_only=True) > 0
