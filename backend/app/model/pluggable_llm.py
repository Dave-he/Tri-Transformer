from typing import Optional

import torch
import torch.nn as nn

from app.model.lora_adapter import LoraAdapter


class PluggableLLMAdapter(nn.Module):
    def __init__(
        self,
        branch: nn.Module,
        external_layers: nn.Module,
        inject_lora: bool = False,
        lora_rank: int = 8,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.branch = branch
        self.external_layers = external_layers
        self.inject_lora = inject_lora

        if inject_lora:
            self._inject_lora_into(self.branch, lora_rank, freeze_base)
            self._inject_lora_into(self.external_layers, lora_rank, freeze_base)

    def _inject_lora_into(self, module: nn.Module, rank: int, freeze_base: bool):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoraAdapter(child, rank=rank, freeze_base=freeze_base))
            else:
                self._inject_lora_into(child, rank, freeze_base)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.branch(src, src_key_padding_mask=src_key_padding_mask)
