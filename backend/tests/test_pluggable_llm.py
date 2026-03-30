import math
import pytest
import torch
import torch.nn as nn


class TestLoraAdapter:
    def test_output_shape(self):
        from app.model.lora_adapter import LoraAdapter

        linear = nn.Linear(64, 64)
        lora = LoraAdapter(linear, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 64)

    def test_freeze_base_true(self):
        from app.model.lora_adapter import LoraAdapter

        linear = nn.Linear(64, 64)
        lora = LoraAdapter(linear, rank=4, freeze_base=True)
        assert not lora.base_weight.requires_grad
        assert not lora.base_bias.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_freeze_base_false(self):
        from app.model.lora_adapter import LoraAdapter

        linear = nn.Linear(64, 64)
        lora = LoraAdapter(linear, rank=4, freeze_base=False)
        assert lora.base_weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_param_count_less_than_base(self):
        from app.model.lora_adapter import LoraAdapter

        in_f, out_f, rank = 1024, 1024, 8
        linear = nn.Linear(in_f, out_f)
        lora = LoraAdapter(linear, rank=rank, freeze_base=True)

        base_params = in_f * out_f + out_f
        lora_params = sum(p.numel() for p in [lora.lora_A, lora.lora_B])
        assert lora_params < base_params * 0.05

    def test_lora_b_initialized_zero(self):
        from app.model.lora_adapter import LoraAdapter

        linear = nn.Linear(64, 64)
        lora = LoraAdapter(linear, rank=4)
        assert torch.all(lora.lora_B == 0).item()

    def test_gradient_flows_through_lora(self):
        from app.model.lora_adapter import LoraAdapter

        linear = nn.Linear(32, 32)
        lora = LoraAdapter(linear, rank=4, freeze_base=True)
        x = torch.randn(1, 32, requires_grad=False)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.grad is not None


class TestPluggableLLMAdapter:
    def _make_mock_encoder(self, d_model=64):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=2)

    def test_forward_output_shape(self):
        from app.model.pluggable_llm import PluggableLLMAdapter
        from app.model.branches import ITransformer

        external_encoder = self._make_mock_encoder(d_model=64)
        branch = ITransformer(vocab_size=100, d_model=64, num_heads=4,
                              num_layers=2, dim_feedforward=128, dropout=0.0)
        adapter = PluggableLLMAdapter(branch=branch, external_layers=external_encoder)

        src = torch.randint(1, 100, (2, 10))
        out = adapter(src)
        assert out.shape == (2, 10, 64)

    def test_inject_lora_wraps_linears(self):
        from app.model.pluggable_llm import PluggableLLMAdapter
        from app.model.lora_adapter import LoraAdapter
        from app.model.branches import ITransformer

        external_encoder = self._make_mock_encoder(d_model=64)
        branch = ITransformer(vocab_size=100, d_model=64, num_heads=4,
                              num_layers=2, dim_feedforward=128, dropout=0.0)
        adapter = PluggableLLMAdapter(branch=branch, external_layers=external_encoder,
                                      inject_lora=True, lora_rank=4)

        lora_count = sum(1 for m in adapter.modules() if isinstance(m, LoraAdapter))
        assert lora_count > 0

    def test_gradient_backprop(self):
        from app.model.pluggable_llm import PluggableLLMAdapter
        from app.model.branches import ITransformer

        external_encoder = self._make_mock_encoder(d_model=64)
        branch = ITransformer(vocab_size=100, d_model=64, num_heads=4,
                              num_layers=2, dim_feedforward=128, dropout=0.0)
        adapter = PluggableLLMAdapter(branch=branch, external_layers=external_encoder)

        src = torch.randint(1, 100, (2, 5))
        out = adapter(src)
        loss = out.sum()
        loss.backward()

        grads = [p.grad for p in adapter.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0
