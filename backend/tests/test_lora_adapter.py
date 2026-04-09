import pytest
import torch
import torch.nn as nn

from app.model.lora_adapter import DoraAdapter, LoraAdapter


def make_linear(in_f=16, out_f=32, bias=True):
    lin = nn.Linear(in_f, out_f, bias=bias)
    nn.init.normal_(lin.weight)
    if bias:
        nn.init.normal_(lin.bias)
    return lin


class TestDoraAdapter:
    def test_forward_shape(self):
        lin = make_linear(16, 32)
        dora = DoraAdapter(lin, rank=4)
        x = torch.randn(2, 16)
        out = dora(x)
        assert out.shape == (2, 32)

    def test_magnitude_vector_shape(self):
        lin = make_linear(16, 32)
        dora = DoraAdapter(lin, rank=4)
        assert dora.magnitude.shape == (32,)

    def test_direction_normalized(self):
        lin = make_linear(16, 32)
        dora = DoraAdapter(lin, rank=4)
        W = dora._effective_weight()
        norms = W.norm(dim=1)
        assert torch.allclose(norms, dora.magnitude, atol=1e-5)

    def test_freeze_base_by_default(self):
        lin = make_linear(16, 32)
        dora = DoraAdapter(lin, rank=4)
        assert not dora.base_weight.requires_grad

    def test_lora_params_trainable(self):
        lin = make_linear(16, 32)
        dora = DoraAdapter(lin, rank=4)
        assert dora.lora_A.requires_grad
        assert dora.lora_B.requires_grad
        assert dora.magnitude.requires_grad

    def test_gradient_flows_through_lora(self):
        lin = make_linear(8, 16)
        dora = DoraAdapter(lin, rank=4)
        x = torch.randn(2, 8)
        out = dora(x)
        loss = out.sum()
        loss.backward()
        assert dora.lora_A.grad is not None
        assert dora.lora_B.grad is not None
        assert dora.magnitude.grad is not None

    def test_no_bias(self):
        lin = make_linear(8, 16, bias=False)
        dora = DoraAdapter(lin, rank=4)
        assert dora.bias is None
        x = torch.randn(2, 8)
        out = dora(x)
        assert out.shape == (2, 16)

    def test_alpha_scaling(self):
        lin = make_linear(8, 16)
        dora_r4 = DoraAdapter(lin, rank=4, alpha=8.0)
        assert abs(dora_r4.scaling - 2.0) < 1e-6

    def test_differs_from_lora_after_update(self):
        torch.manual_seed(0)
        lin = make_linear(8, 16)
        lora = LoraAdapter(lin, rank=4)
        dora = DoraAdapter(lin, rank=4)
        x = torch.randn(2, 8)
        nn.init.normal_(dora.lora_B, std=0.1)
        nn.init.normal_(lora.lora_B, std=0.1)
        with torch.no_grad():
            out_lora = lora(x)
            out_dora = dora(x)
        assert not torch.allclose(out_lora, out_dora, atol=1e-4)


class TestLoraAdapterLRRatio:
    def test_lr_ratio_param_groups(self):
        lin = make_linear(16, 32)
        adapter = LoraAdapter(lin, rank=8, lr_ratio=4.0)
        groups = adapter.param_groups(base_lr=1e-3)
        lrs = {g["name"]: g["lr"] for g in groups}
        assert abs(lrs["lora_A"] - 1e-3 / 4.0) < 1e-10
        assert abs(lrs["lora_B"] - 1e-3) < 1e-10

    def test_lr_ratio_default_1(self):
        lin = make_linear(16, 32)
        adapter = LoraAdapter(lin, rank=8)
        groups = adapter.param_groups(base_lr=1e-3)
        lrs = {g["name"]: g["lr"] for g in groups}
        assert abs(lrs["lora_A"] - 1e-3) < 1e-10

    def test_lr_ratio_16(self):
        lin = make_linear(16, 32)
        adapter = LoraAdapter(lin, rank=8, lr_ratio=16.0)
        groups = adapter.param_groups(base_lr=1e-3)
        lrs = {g["name"]: g["lr"] for g in groups}
        assert abs(lrs["lora_A"] - 1e-3 / 16.0) < 1e-10

    def test_forward_unchanged(self):
        lin = make_linear(8, 16)
        adapter = LoraAdapter(lin, rank=4, lr_ratio=8.0)
        x = torch.randn(2, 8)
        out = adapter(x)
        assert out.shape == (2, 16)
