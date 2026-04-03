import torch
import torch.nn as nn


class LoraAdapter(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = None,
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
        _alpha = alpha if alpha is not None else rank
        self.scaling = _alpha / rank

        nn.init.normal_(self.lora_A, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.base_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.base_weight, self.base_bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out
