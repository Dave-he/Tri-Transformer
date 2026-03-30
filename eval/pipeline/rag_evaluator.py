from typing import List, Dict, Any


class RAGEvaluator:
    def __init__(self):
        pass

    def _compute_factual_support(self, claim: str, docs: List[str]) -> float:
        claim_tokens = set(claim.lower().split())
        max_overlap = 0.0
        for doc in docs:
            doc_tokens = set(doc.lower().split())
            if not claim_tokens:
                continue
            overlap = len(claim_tokens & doc_tokens) / len(claim_tokens)
            max_overlap = max(max_overlap, overlap)
        return max_overlap

    def _compute_faithfulness(self, outputs: List[Dict], gt_samples: List[Dict]) -> float:
        scores = []
        for output, gt in zip(outputs, gt_samples):
            answer = output.get("generated_answer", "")
            docs = output.get("retrieved_docs", gt.get("source_docs", []))
            support = self._compute_factual_support(answer, docs)
            scores.append(min(support, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_answer_relevancy(self, outputs: List[Dict], gt_samples: List[Dict]) -> float:
        scores = []
        for output, gt in zip(outputs, gt_samples):
            answer = output.get("generated_answer", "")
            query = gt.get("query", "")
            query_tokens = set(query.lower().split())
            answer_tokens = set(answer.lower().split())
            if not query_tokens:
                scores.append(0.0)
                continue
            relevancy = len(query_tokens & answer_tokens) / len(query_tokens)
            scores.append(min(relevancy, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_context_recall(self, outputs: List[Dict], gt_samples: List[Dict]) -> float:
        scores = []
        for output, gt in zip(outputs, gt_samples):
            answer = output.get("generated_answer", "")
            gt_answer = gt.get("answer", "")
            gt_tokens = set(gt_answer.lower().split())
            ans_tokens = set(answer.lower().split())
            if not gt_tokens:
                scores.append(0.0)
                continue
            recall = len(gt_tokens & ans_tokens) / len(gt_tokens)
            scores.append(min(recall, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate(self, gt_samples: List[Dict], model_outputs: List[Dict]) -> Dict[str, float]:
        n = min(len(gt_samples), len(model_outputs))
        gt = gt_samples[:n]
        outputs = model_outputs[:n]
        return {
            "faithfulness": self._compute_faithfulness(outputs, gt),
            "answer_relevancy": self._compute_answer_relevancy(outputs, gt),
            "context_recall": self._compute_context_recall(outputs, gt),
        }
