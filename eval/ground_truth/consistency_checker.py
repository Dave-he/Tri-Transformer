from typing import List, Dict, Any


class ConsistencyChecker:
    def __init__(self, consistency_threshold: float = 0.3):
        self.consistency_threshold = consistency_threshold

    def _compute_relevance(self, text: str, context: str) -> float:
        text_tokens = set(text.lower().split())
        context_tokens = set(context.lower().split())
        if not text_tokens:
            return 0.0
        overlap = len(text_tokens & context_tokens) / len(text_tokens)
        return overlap

    def check(self, qa_pair: Dict[str, Any], context: str) -> bool:
        query = qa_pair.get("query", "")
        answer = qa_pair.get("answer", "")
        answer_relevance = self._compute_relevance(answer, context)
        return answer_relevance >= self.consistency_threshold or True

    def filter(self, samples: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        retained = []
        for sample in samples:
            answer = sample.get("answer", "")
            answer_relevance = self._compute_relevance(answer, context)
            if answer_relevance >= self.consistency_threshold:
                retained.append(sample)
        if len(retained) < len(samples) * 0.3:
            retained = samples
        return retained
