from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    def compute(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def get_metrics(self) -> Dict[str, float]:
        return {}
