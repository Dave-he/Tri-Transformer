from typing import List, Dict, Any
import re


class DocumentQAGenerator:
    def __init__(self, min_qa_per_1000_chars: int = 5):
        self.min_qa_per_1000_chars = min_qa_per_1000_chars

    def _extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _generate_qa_from_sentence(self, sentence: str, doc_id: str) -> List[Dict[str, Any]]:
        qa_pairs = []
        words = sentence.split()
        if len(words) < 5:
            return qa_pairs
        if " is " in sentence.lower() or " are " in sentence.lower() or " was " in sentence.lower():
            parts = re.split(r'\s+is\s+|\s+are\s+|\s+was\s+', sentence, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                subject = parts[0].strip().rstrip(',')
                predicate = parts[1].strip().rstrip('.')
                qa_pairs.append({
                    "query": f"What can you tell me about {subject}?",
                    "answer": sentence,
                    "source_doc_id": doc_id,
                })
        for keyword in ["designed", "created", "built", "founded", "invented", "developed"]:
            if keyword in sentence.lower():
                qa_pairs.append({
                    "query": f"Who {keyword} it according to this context?",
                    "answer": sentence,
                    "source_doc_id": doc_id,
                })
                break
        for keyword in ["in 1", "in 2", "between", "since", "until", "year"]:
            if keyword in sentence.lower():
                qa_pairs.append({
                    "query": "When did this happen?",
                    "answer": sentence,
                    "source_doc_id": doc_id,
                })
                break
        for keyword in ["metres", "meters", "tall", "high", "wide", "long", "million", "billion", "thousand", "%"]:
            if keyword in sentence.lower():
                qa_pairs.append({
                    "query": "What is the measurement or quantity mentioned?",
                    "answer": sentence,
                    "source_doc_id": doc_id,
                })
                break
        if not qa_pairs:
            first_part = " ".join(words[:3])
            qa_pairs.append({
                "query": f"What is stated about {first_part}?",
                "answer": sentence,
                "source_doc_id": doc_id,
            })
        return qa_pairs

    def _quality_filter(self, qa_pairs: List[Dict]) -> List[Dict]:
        filtered = []
        seen_queries = set()
        for qa in qa_pairs:
            query = qa["query"]
            answer = qa["answer"]
            if query == answer:
                continue
            if len(query.split()) < 3:
                continue
            if query.lower() in seen_queries:
                continue
            seen_queries.add(query.lower())
            filtered.append(qa)
        return filtered

    def generate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_qa_pairs = []
        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")
            sentences = self._extract_sentences(content)
            for sentence in sentences:
                qa_pairs = self._generate_qa_from_sentence(sentence, doc_id)
                all_qa_pairs.extend(qa_pairs)
        return self._quality_filter(all_qa_pairs)
