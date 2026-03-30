from typing import List, Dict, Any


class HallucinationEvaluator:
    def __init__(self, hallucination_threshold: float = 0.3):
        self.hallucination_threshold = hallucination_threshold

    def _compute_factual_support(self, text: str, docs: List[str]) -> float:
        text_tokens = set(text.lower().split())
        max_overlap = 0.0
        for doc in docs:
            doc_tokens = set(doc.lower().split())
            if not text_tokens:
                continue
            overlap = len(text_tokens & doc_tokens) / len(text_tokens)
            max_overlap = max(max_overlap, overlap)
        return max_overlap

    def compute_hallucination_rate(self, outputs: List[Dict]) -> float:
        if not outputs:
            return 0.0
        hallucination_count = 0
        for output in outputs:
            answer = output.get("generated_answer", "")
            docs = output.get("retrieved_docs", [])
            support = self._compute_factual_support(answer, docs)
            if support < self.hallucination_threshold:
                hallucination_count += 1
        return hallucination_count / len(outputs)

    def compute_source_attribution_rate(self, outputs: List[Dict]) -> float:
        scores = []
        for output in outputs:
            answer = output.get("generated_answer", "")
            docs = output.get("retrieved_docs", [])
            score = self._compute_factual_support(answer, docs)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate(self, gt_samples: List[Dict], model_outputs: List[Dict]) -> Dict[str, float]:
        return {
            "hallucination_rate": self.compute_hallucination_rate(model_outputs),
            "source_attribution_rate": self.compute_source_attribution_rate(model_outputs),
        }
