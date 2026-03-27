from typing import List, Dict, Any


class ControlEvaluator:
    def _instruction_following_rate(self, outputs: List[Dict], instructions: List[str]) -> float:
        scores = []
        for output, instruction in zip(outputs, instructions):
            answer = output.get("generated_answer", "")
            inst_tokens = set(instruction.lower().split())
            ans_tokens = set(answer.lower().split())
            overlap = len(inst_tokens & ans_tokens) / max(len(inst_tokens), 1)
            scores.append(min(overlap, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def _topic_consistency(self, outputs: List[Dict], gt_samples: List[Dict]) -> float:
        scores = []
        for output, gt in zip(outputs, gt_samples):
            answer = output.get("generated_answer", "")
            query = gt.get("query", "")
            query_tokens = set(query.lower().split())
            ans_tokens = set(answer.lower().split())
            if not query_tokens:
                scores.append(0.0)
                continue
            consistency = len(query_tokens & ans_tokens) / len(query_tokens)
            scores.append(min(consistency, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def evaluate(self, gt_samples: List[Dict], model_outputs: List[Dict]) -> Dict[str, float]:
        instructions = [gt.get("query", "") for gt in gt_samples]
        return {
            "instruction_following_rate": self._instruction_following_rate(model_outputs, instructions),
            "topic_consistency": self._topic_consistency(model_outputs, gt_samples),
        }
