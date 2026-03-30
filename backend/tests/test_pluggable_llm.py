import pytest
import torch
import torch.nn as nn
from unittest.mock import patch


class TestLoraAdapter:
    def test_lora_adapter_forward_output_shape(self):
        from app.model.lora_adapter import LoraAdapter
        linear = nn.Linear(64, 64)
        adapter = LoraAdapter(linear, rank=4, alpha=8)
        x = torch.randn(2, 10, 64)
        output = adapter(x)
        assert output.shape == x.shape

    def test_lora_adapter_frozen_base(self):
        from app.model.lora_adapter import LoraAdapter
        linear = nn.Linear(64, 64)
        adapter = LoraAdapter(linear, rank=4, alpha=8, freeze_base=True)
        for name, param in adapter.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_lora_adapter_trainable_params(self):
        from app.model.lora_adapter import LoraAdapter
        linear = nn.Linear(64, 64)
        adapter = LoraAdapter(linear, rank=4, alpha=8)
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        base_params = sum(p.numel() for p in linear.parameters())
        assert trainable_params < base_params * 0.05


class TestPluggableLLMAdapter:
    def test_pluggable_adapter_forward_shape(self):
        from app.model.pluggable_llm import PluggableLLMAdapter, TransformerEncoder
        encoder = TransformerEncoder(d_model=64, nhead=2, num_layers=1, dim_feedforward=128)
        adapter = PluggableLLMAdapter(encoder, d_model=64)
        x = torch.randn(2, 10, 64)
        output = adapter(x)
        assert output.shape == x.shape

    def test_pluggable_adapter_inject_lora(self):
        from app.model.pluggable_llm import PluggableLLMAdapter, TransformerEncoder
        encoder = TransformerEncoder(d_model=64, nhead=2, num_layers=1, dim_feedforward=128)
        adapter = PluggableLLMAdapter(encoder, d_model=64)
        adapter.inject_lora(rank=4, alpha=8)
        assert hasattr(adapter, "_lora_modules")
        x = torch.randn(2, 10, 64)
        output = adapter(x)
        assert output.shape == x.shape

    def test_pluggable_adapter_backward(self):
        from app.model.pluggable_llm import PluggableLLMAdapter, TransformerEncoder
        encoder = TransformerEncoder(d_model=64, nhead=2, num_layers=1, dim_feedforward=128)
        adapter = PluggableLLMAdapter(encoder, d_model=64)
        adapter.inject_lora(rank=4, alpha=8)
        x = torch.randn(2, 10, 64, requires_grad=True)
        output = adapter(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
