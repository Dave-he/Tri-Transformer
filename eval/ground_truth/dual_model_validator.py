from typing import List, Tuple


class DualModelValidator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def is_consensus(self, answer_a: str, answer_b: str) -> bool:
        sim = self._compute_similarity(answer_a, answer_b)
        return sim >= self.similarity_threshold

    def get_quality_label(self, answer_a: str, answer_b: str) -> str:
        sim = self._compute_similarity(answer_a, answer_b)
        if sim >= self.similarity_threshold:
            return "gold"
        elif sim >= 0.5:
            return "silver"
        else:
            return "discard"
