import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraAdapter(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = None,
        freeze_base: bool = True,
        lr_ratio: float = 1.0,
    ):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.lr_ratio = lr_ratio

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
        base_out = F.linear(x, self.base_weight, self.base_bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def param_groups(self, base_lr: float = 1e-3):
        return [
            {"name": "lora_A", "params": [self.lora_A], "lr": base_lr / self.lr_ratio},
            {"name": "lora_B", "params": [self.lora_B], "lr": base_lr},
        ]


class DoraAdapter(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = None,
        freeze_base: bool = True,
        lr_ratio: float = 1.0,
    ):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.lr_ratio = lr_ratio

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

        with torch.no_grad():
            W = self.base_weight + (self.lora_B @ self.lora_A) * self.scaling
            m = W.norm(dim=1)
        self.magnitude = nn.Parameter(m.clone())

    @property
    def bias(self):
        return self.base_bias

    def _effective_weight(self) -> torch.Tensor:
        W = self.base_weight + (self.lora_B @ self.lora_A) * self.scaling
        col_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        direction = W / col_norms
        return direction * self.magnitude.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._effective_weight()
        return F.linear(x, W, self.bias)

    def param_groups(self, base_lr: float = 1e-3):
        return [
            {"name": "lora_A", "params": [self.lora_A], "lr": base_lr / self.lr_ratio},
            {"name": "lora_B", "params": [self.lora_B], "lr": base_lr},
            {"name": "magnitude", "params": [self.magnitude], "lr": base_lr},
        ]
