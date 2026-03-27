from typing import List, Dict, Any
import re


class DialogCohesionEvaluator:
    def _compute_turn_consistency(self, turns: List[str]) -> float:
        if len(turns) < 2:
            return 1.0
        scores = []
        for i in range(1, len(turns)):
            prev_tokens = set(re.findall(r'\w+', turns[i - 1].lower()))
            curr_tokens = set(re.findall(r'\w+', turns[i].lower()))
            if not prev_tokens:
                scores.append(0.0)
                continue
            overlap = len(prev_tokens & curr_tokens) / len(prev_tokens)
            scores.append(min(overlap, 1.0))
        return sum(scores) / len(scores) if scores else 1.0

    def evaluate(self, dialog_sessions: List[List[Dict]]) -> Dict[str, float]:
        if not dialog_sessions:
            return {"multi_turn_consistency": 1.0, "context_retention_rate": 1.0}
        consistencies = []
        for session in dialog_sessions:
            turns = [turn.get("text", turn.get("content", "")) for turn in session]
            consistencies.append(self._compute_turn_consistency(turns))
        return {
            "multi_turn_consistency": sum(consistencies) / len(consistencies),
            "context_retention_rate": sum(consistencies) / len(consistencies),
        }
