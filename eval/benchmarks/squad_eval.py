from __future__ import annotations

import json
import re
from typing import Any, Callable

from eval.benchmarks.base import BaseBenchmarkEvaluator, BenchmarkResult


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class SQuADEvaluator(BaseBenchmarkEvaluator):
    """
    SQuAD 2.0 阅读理解基准评测。
    期望 model_fn(context, question) -> str。
    数据集格式: list of {"context": str, "question": str, "answers": list[str], "is_impossible": bool}
    """

    name = "SQuAD-2.0"

    def __init__(self, dataset_path: str = "eval/benchmarks/data/squad_sample.jsonl"):
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

    def evaluate(self, model_fn: Callable[[str, str], str]) -> BenchmarkResult:
        samples = self.load_dataset()
        if not samples:
            return BenchmarkResult(benchmark=self.name, metrics={"em": 0.0, "f1": 0.0}, num_samples=0)

        total_em = 0.0
        total_f1 = 0.0
        for sample in samples:
            pred = model_fn(sample["context"], sample["question"])
            answers = sample.get("answers", [])
            if sample.get("is_impossible") and not pred.strip():
                total_em += 1.0
                total_f1 += 1.0
                continue
            best_em = max((_normalize(pred) == _normalize(a) for a in answers), default=False)
            best_f1 = max((_token_f1(pred, a) for a in answers), default=0.0)
            total_em += float(best_em)
            total_f1 += best_f1

        n = len(samples)
        return BenchmarkResult(
            benchmark=self.name,
            metrics={"em": total_em / n, "f1": total_f1 / n},
            num_samples=n,
        )
