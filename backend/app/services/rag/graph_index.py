import json
import os
import re
from typing import Optional


def _extract_terms(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())
    stopwords = {
        "the", "and", "for", "are", "was", "with", "that", "this",
        "have", "from", "not", "but", "they", "been", "its", "which",
        "can", "all", "also", "over", "use", "used", "using",
    }
    return [t for t in tokens if t not in stopwords]


def _detect_communities_greedy(adjacency: dict) -> list[list[str]]:
    nodes = set(adjacency.keys())
    visited = set()
    communities = []
    for node in nodes:
        if node in visited:
            continue
        component = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            visited.add(cur)
            for nbr in adjacency.get(cur, []):
                if nbr not in component:
                    stack.append(nbr)
        communities.append(list(component))
    return communities


class GraphRAGIndex:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self._docs: dict[str, str] = {}
        self._communities: list[list[str]] = []
        self._community_summaries: list[dict] = []
        self._graph: dict[str, list[str]] = {}

    def _build_graph(self):
        term_to_docs: dict[str, list[str]] = {}
        for doc_id, text in self._docs.items():
            for term in set(_extract_terms(text)):
                term_to_docs.setdefault(term, []).append(doc_id)
        adj: dict[str, list[str]] = {doc_id: [] for doc_id in self._docs}
        for docs in term_to_docs.values():
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    a, b = docs[i], docs[j]
                    if b not in adj[a]:
                        adj[a].append(b)
                    if a not in adj[b]:
                        adj[b].append(a)
        self._graph = adj

    def _build_communities(self):
        self._build_graph()
        raw = _detect_communities_greedy(self._graph)
        self._communities = raw
        self._community_summaries = []
        for comm in raw:
            texts = [self._docs[d] for d in comm if d in self._docs]
            merged = " ".join(texts)
            terms = _extract_terms(merged)
            freq: dict[str, int] = {}
            for t in terms:
                freq[t] = freq.get(t, 0) + 1
            top_terms = sorted(freq.items(), key=lambda x: -x[1])[:10]
            summary_text = "; ".join(f"{k}({v})" for k, v in top_terms)
            self._community_summaries.append({
                "doc_ids": comm,
                "summary": summary_text,
                "top_terms": [k for k, _ in top_terms],
            })

    def add_documents(self, docs: list[dict]):
        for d in docs:
            self._docs[d["id"]] = d["text"]
        if self._docs:
            self._build_communities()

    def remove_document(self, doc_id: str):
        self._docs.pop(doc_id, None)
        if self._docs:
            self._build_communities()
        else:
            self._communities = []
            self._community_summaries = []
            self._graph = {}

    def global_query(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._community_summaries:
            return []
        query_terms = set(_extract_terms(query))
        scored = []
        for comm_data in self._community_summaries:
            top_terms = set(comm_data.get("top_terms", []))
            overlap = len(query_terms & top_terms)
            score = overlap / (len(query_terms) + 1e-8)
            scored.append({
                "summary": comm_data["summary"],
                "doc_ids": comm_data["doc_ids"],
                "score": score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def community_count(self) -> int:
        return len(self._communities)

    def doc_count(self) -> int:
        return len(self._docs)

    def save(self):
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, "graph_rag_index.json")
        data = {
            "docs": self._docs,
            "community_summaries": self._community_summaries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self):
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "graph_rag_index.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._docs = data.get("docs", {})
        self._community_summaries = data.get("community_summaries", [])
        self._communities = [c["doc_ids"] for c in self._community_summaries]
        if self._docs:
            self._build_graph()
