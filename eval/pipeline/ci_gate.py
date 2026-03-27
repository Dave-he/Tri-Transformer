from typing import List, Dict, Any, Tuple


class CIGate:
    def __init__(
        self,
        hallucination_rate_threshold: float = 0.05,
        rag_recall_at_5_threshold: float = 0.90,
        bert_score_f1_threshold: float = 0.85,
    ):
        self.hallucination_rate_threshold = hallucination_rate_threshold
        self.rag_recall_at_5_threshold = rag_recall_at_5_threshold
        self.bert_score_f1_threshold = bert_score_f1_threshold

    def check(self, report: Dict[str, float]) -> Tuple[bool, str]:
        failures = []
        hall_rate = report.get("hallucination_rate", 0.0)
        if hall_rate >= self.hallucination_rate_threshold:
            failures.append(
                f"hallucination_rate {hall_rate:.3f} >= threshold {self.hallucination_rate_threshold}"
            )
        recall_5 = report.get("rag_recall_at_5", 1.0)
        if recall_5 <= self.rag_recall_at_5_threshold:
            failures.append(
                f"rag_recall_at_5 {recall_5:.3f} <= threshold {self.rag_recall_at_5_threshold}"
            )
        bert_f1 = report.get("bert_score_f1", 1.0)
        if bert_f1 <= self.bert_score_f1_threshold:
            failures.append(
                f"bert_score_f1 {bert_f1:.3f} <= threshold {self.bert_score_f1_threshold}"
            )
        if failures:
            return False, "CI GATE FAILED:\n" + "\n".join(failures)
        return True, ""
