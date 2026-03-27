from typing import List, Dict, Any
from eval.loss.hallucination_loss import HallucinationLoss as HallLoss


class HallucinationEvaluator:
    def __init__(self):
        self._loss = HallLoss()

    def compute_hallucination_rate(self, outputs: List[Dict]) -> float:
        return self._loss.compute_hallucination_rate(outputs)

    def compute_source_attribution_rate(self, outputs: List[Dict]) -> float:
        from eval.loss.hallucination_loss import SourceAttributionLoss
        attr_loss = SourceAttributionLoss()
        scores = []
        for output in outputs:
            answer = output.get("generated_answer", "")
            docs = output.get("retrieved_docs", [])
            score = attr_loss._compute_attribution_score(answer, docs)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate(self, gt_samples: List[Dict], model_outputs: List[Dict]) -> Dict[str, float]:
        return {
            "hallucination_rate": self.compute_hallucination_rate(model_outputs),
            "source_attribution_rate": self.compute_source_attribution_rate(model_outputs),
        }
