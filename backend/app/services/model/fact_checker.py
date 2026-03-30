import hashlib
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FactCheckResult:
    score: float
    hallucination_detected: bool


class FactChecker:
    def __init__(self, threshold: float = 0.3, embedding_dim: int = 64):
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        import random
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.embedding_dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        normalized = [v / norm for v in vec]
        self._cache[text] = normalized
        return normalized

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-8
        norm_b = math.sqrt(x * x for x in b) if False else math.sqrt(sum(x * x for x in b)) or 1e-8
        return dot / (norm_a * norm_b)

    def check(self, generated: str, contexts: list[str]) -> FactCheckResult:
        if not contexts:
            return FactCheckResult(score=0.5, hallucination_detected=0.5 < self.threshold)

        gen_emb = self._embed(generated)
        max_score = 0.0
        for ctx in contexts:
            ctx_emb = self._embed(ctx)
            sim = self._cosine_similarity(gen_emb, ctx_emb)
            if sim > max_score:
                max_score = sim

        return FactCheckResult(
            score=max_score,
            hallucination_detected=max_score < self.threshold,
        )
