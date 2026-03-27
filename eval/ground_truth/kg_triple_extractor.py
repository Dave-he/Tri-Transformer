from typing import List, Tuple, Dict, Any
import re


class KGTripleExtractor:
    def __init__(self):
        self._verb_patterns = [
            r'(\w[\w\s]+?)\s+(is|are|was|were|has|have|had|became|designed|created|built|founded|invented|located)\s+([\w][\w\s,]+)',
        ]

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        triples = []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in sentences:
            for pattern in self._verb_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    subject = match.group(1).strip()
                    relation = match.group(2).strip()
                    obj = match.group(3).strip().rstrip('.,;')
                    if len(subject.split()) <= 6 and len(obj.split()) <= 8:
                        triples.append((subject, relation, obj))
        return triples

    def triple_to_qa(self, triple: Tuple[str, str, str]) -> Dict[str, Any]:
        subject, relation, obj = triple
        relation_map = {
            "is": "What is",
            "are": "What are",
            "was": "What was",
            "were": "What were",
            "has": "What does have",
            "have": "What does have",
            "located": "Where is",
            "designed": "Who designed",
            "created": "Who created",
            "built": "Who built",
            "founded": "Who founded",
        }
        rel_phrase = relation_map.get(relation.lower(), f"What is the {relation} of")
        query = f"{rel_phrase} {subject}?"
        answer = f"{subject} {relation} {obj}."
        return {
            "query": query,
            "answer": answer,
            "triple": (subject, relation, obj),
        }
