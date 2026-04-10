from __future__ import annotations

import json
from typing import Any, Callable

from eval.benchmarks.base import BaseBenchmarkEvaluator, BenchmarkResult


class TruthfulQAEvaluator(BaseBenchmarkEvaluator):
    """
    TruthfulQA 幻觉检测基准评测。
    期望 model_fn(question) -> str。
    数据集格式: list of {"question": str, "correct_answers": list[str], "incorrect_answers": list[str]}
    """

    name = "TruthfulQA"

    def __init__(self, dataset_path: str = "eval/benchmarks/data/truthfulqa_sample.jsonl"):
        self.dataset_path = dataset_path

    def load_dataset(self) -> list[dict[str, Any]]:
        samples = []
        try:
            with open(self.dataset_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except FileNotFoundError:
            pass
        return samples

    def _is_truthful(self, pred: str, correct_answers: list[str], incorrect_answers: list[str]) -> bool:
        pred_lower = pred.lower()
        for ans in correct_answers:
            if ans.lower() in pred_lower:
                return True
        for ans in incorrect_answers:
            if ans.lower() in pred_lower:
                return False
        return False

    def evaluate(self, model_fn: Callable[[str], str]) -> BenchmarkResult:
        samples = self.load_dataset()
        if not samples:
            return BenchmarkResult(benchmark=self.name, metrics={"truthful_rate": 0.0}, num_samples=0)

        truthful = 0
        for sample in samples:
            pred = model_fn(sample["question"])
            if self._is_truthful(pred, sample.get("correct_answers", []), sample.get("incorrect_answers", [])):
                truthful += 1

        n = len(samples)
        return BenchmarkResult(
            benchmark=self.name,
            metrics={"truthful_rate": truthful / n},
            num_samples=n,
        )
