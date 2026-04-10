#!/usr/bin/env python3
"""一键运行所有基准评测并输出对比报告。"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eval.benchmarks.squad_eval import SQuADEvaluator
from eval.benchmarks.truthfulqa_eval import TruthfulQAEvaluator
from eval.benchmarks.hotpotqa_eval import HotpotQAEvaluator


def mock_qa_model(context: str, question: str) -> str:
    return ""


def mock_truthful_model(question: str) -> str:
    return ""


def mock_multihop_model(contexts: list[str], question: str) -> str:
    return ""


def run_all():
    evaluators = [
        (SQuADEvaluator(), mock_qa_model),
        (TruthfulQAEvaluator(), mock_truthful_model),
        (HotpotQAEvaluator(), mock_multihop_model),
    ]

    print("=" * 60)
    print("Tri-Transformer 基准评测报告")
    print("=" * 60)
    for evaluator, model_fn in evaluators:
        result = evaluator.evaluate(model_fn)
        print(evaluator.report(result))
        print()


if __name__ == "__main__":
    run_all()
