from typing import List, Dict, Any, Union
from eval.ground_truth.schema import GroundTruthItem, SourceType


class GTFusionEngine:
    SOURCE_WEIGHTS = {
        SourceType.HUMAN: 1.0,
        SourceType.DUAL_MODEL: 0.85,
        SourceType.DOCUMENT_QA: 0.6,
        SourceType.KG_TRIPLE: 0.65,
    }

    def _classify_difficulty(self, query: str, answer: str) -> str:
        query_words = len(query.split())
        answer_words = len(answer.split())
        if query_words <= 8 and answer_words <= 15:
            return "easy"
        elif query_words <= 15 or answer_words <= 30:
            return "medium"
        else:
            return "hard"

    def _compute_quality_score(self, sample: Dict[str, Any]) -> float:
        source_type = sample.get("source_type", SourceType.DOCUMENT_QA)
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type)
            except ValueError:
                source_type = SourceType.DOCUMENT_QA
        base_weight = self.SOURCE_WEIGHTS.get(source_type, 0.6)
        query = sample.get("query", "")
        answer = sample.get("answer", "")
        length_bonus = min(len(answer.split()) / 20.0, 0.2)
        return min(base_weight + length_bonus, 1.0)

    def fuse(self, raw_samples: List[Dict[str, Any]]) -> List[GroundTruthItem]:
        fused = []
        for i, sample in enumerate(raw_samples):
            query = sample.get("query", "")
            answer = sample.get("answer", "")
            source_type = sample.get("source_type", SourceType.DOCUMENT_QA)
            difficulty = self._classify_difficulty(query, answer)
            quality_score = self._compute_quality_score(sample)
            item = GroundTruthItem(
                id=sample.get("id", f"fused_{i}"),
                query=query,
                answer=answer,
                source_docs=sample.get("source_docs", [sample.get("source_doc_id", "")]),
                difficulty=difficulty,
                quality_score=quality_score,
                source_type=source_type,
                metadata=sample.get("metadata", {}),
            )
            fused.append(item)
        return fused
