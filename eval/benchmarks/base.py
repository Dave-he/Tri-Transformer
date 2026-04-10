from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    benchmark: str
    metrics: dict[str, float]
    num_samples: int


class BaseBenchmarkEvaluator(ABC):
    name: str = ""

    @abstractmethod
    def load_dataset(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def evaluate(self, model_fn) -> BenchmarkResult:
        pass

    def report(self, result: BenchmarkResult) -> str:
        lines = [f"=== {result.benchmark} ({result.num_samples} samples) ==="]
        for k, v in result.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)
