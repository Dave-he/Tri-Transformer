from __future__ import annotations

import json
from typing import Any, Callable

from eval.benchmarks.base import BaseBenchmarkEvaluator, BenchmarkResult


class HotpotQAEvaluator(BaseBenchmarkEvaluator):
    """
    HotpotQA 多跳推理基准评测。
    期望 model_fn(contexts, question) -> str，contexts 为 list[str]。
    数据集格式: list of {"question": str, "contexts": list[str], "answer": str}
    """

    name = "HotpotQA"

    def __init__(self, dataset_path: str = "eval/benchmarks/data/hotpotqa_sample.jsonl"):
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

    def evaluate(self, model_fn: Callable[[list[str], str], str]) -> BenchmarkResult:
        samples = self.load_dataset()
        if not samples:
            return BenchmarkResult(benchmark=self.name, metrics={"exact_match": 0.0}, num_samples=0)

        correct = 0
        for sample in samples:
            pred = model_fn(sample.get("contexts", []), sample["question"])
            if pred.strip().lower() == sample["answer"].strip().lower():
                correct += 1

        n = len(samples)
        return BenchmarkResult(
            benchmark=self.name,
            metrics={"exact_match": correct / n},
            num_samples=n,
        )
