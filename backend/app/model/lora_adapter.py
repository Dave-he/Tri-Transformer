import math
import torch
import torch.nn as nn


class LoraAdapter(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        freeze_base: bool = True,
    ):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features

        self.base_weight = nn.Parameter(linear.weight.clone(), requires_grad=not freeze_base)
        if linear.bias is not None:
            self.base_bias = nn.Parameter(linear.bias.clone(), requires_grad=not freeze_base)
        else:
            self.register_parameter("base_bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0 / math.sqrt(rank)

        nn.init.normal_(self.lora_A, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.base_weight, self.base_bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out
