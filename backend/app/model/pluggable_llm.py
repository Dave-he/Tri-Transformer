from typing import Optional

import torch
import torch.nn as nn

from app.model.lora_adapter import LoraAdapter


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class PluggableLLMAdapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self._lora_modules: list[LoraAdapter] = []

    def inject_lora(self, rank: int = 8, alpha: float = None, freeze_base: bool = True):
        self._lora_modules = []
        self._inject_lora_into(self.backbone, rank, alpha, freeze_base)

    def _inject_lora_into(self, module: nn.Module, rank: int, alpha: float, freeze_base: bool):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                lora = LoraAdapter(child, rank=rank, alpha=alpha, freeze_base=freeze_base)
                setattr(module, name, lora)
                self._lora_modules.append(lora)
            else:
                self._inject_lora_into(child, rank, alpha, freeze_base)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, 'forward'):
            try:
                return self.backbone(x, src_key_padding_mask=src_key_padding_mask)
            except TypeError:
                return self.backbone(x)
        return x
