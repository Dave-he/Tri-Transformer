from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from eval.loss.base import BaseLoss


class RetrievalRelevanceLoss(BaseLoss):
    def __init__(self, ks: List[int] = None):
        super().__init__()
        self.ks = ks or [1, 3, 5, 10]

    def compute(self, retrieved_ids: List[List[int]], relevant_ids: List[List[int]]) -> Dict[str, float]:
        metrics = {}
        for k in self.ks:
            total_recall = 0.0
            for ret, rel in zip(retrieved_ids, relevant_ids):
                rel_set = set(rel)
                top_k = set(ret[:k])
                if len(rel_set) == 0:
                    recall = 0.0
                else:
                    recall = len(top_k & rel_set) / len(rel_set)
                total_recall += recall
            metrics[f"recall@{k}"] = total_recall / len(retrieved_ids)
        return metrics

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(1).expand_as(doc_embs),
            doc_embs,
            dim=-1
        )
        top_k = min(5, doc_embs.size(1))
        topk_indices = torch.topk(similarities, top_k, dim=-1).indices
        recall_list = []
        for i in range(labels.size(0)):
            rel_count = labels[i].sum()
            if rel_count == 0:
                recall_list.append(torch.tensor(0.0))
                continue
            retrieved_rel = labels[i][topk_indices[i]].sum()
            recall = retrieved_rel / rel_count
            recall_list.append(recall)
        avg_recall = torch.stack(recall_list).mean()
        return 1.0 - avg_recall


class CoverageLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _token_f1(self, pred: str, gold: str) -> float:
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute(self, predictions: List[str], references: List[str]) -> float:
        f1_scores = [self._token_f1(p, r) for p, r in zip(predictions, references)]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coverage = F.cosine_similarity(
            query_emb.unsqueeze(1).expand_as(doc_embs),
            doc_embs,
            dim=-1
        )
        weighted_coverage = (coverage * labels).sum(dim=-1) / (labels.sum(dim=-1) + 1e-8)
        return 1.0 - weighted_coverage.mean()


class RankingConsistencyLoss(BaseLoss):
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def compute_ndcg(self, scores: List[List[float]], relevance: List[List[float]]) -> float:
        def dcg(sorted_rel, k):
            val = 0.0
            for i, r in enumerate(sorted_rel[:k]):
                val += (2 ** r - 1) / math.log2(i + 2)
            return val

        ndcg_scores = []
        for score_list, rel_list in zip(scores, relevance):
            paired = sorted(zip(score_list, rel_list), key=lambda x: -x[0])
            sorted_rel = [r for _, r in paired]
            ideal_rel = sorted(rel_list, reverse=True)
            actual_dcg = dcg(sorted_rel, self.k)
            ideal_dcg = dcg(ideal_rel, self.k)
            ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    def compute_mrr(self, retrieved_ids: List[List[int]], relevant_ids: List[List[int]]) -> float:
        mrr = 0.0
        for ret, rel in zip(retrieved_ids, relevant_ids):
            rel_set = set(rel)
            for rank, doc_id in enumerate(ret[:self.k], start=1):
                if doc_id in rel_set:
                    mrr += 1.0 / rank
                    break
        return mrr / len(retrieved_ids)

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(1).expand_as(doc_embs),
            doc_embs,
            dim=-1
        )
        sorted_indices = torch.argsort(similarities, dim=-1, descending=True)
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        k = min(self.k, doc_embs.size(1))
        dcg = torch.zeros(labels.size(0))
        ideal_dcg = torch.zeros(labels.size(0))
        for i in range(k):
            discount = 1.0 / math.log2(i + 2)
            dcg += sorted_labels[:, i] * discount
            ideal_sorted = torch.sort(labels, dim=-1, descending=True).values
            ideal_dcg += ideal_sorted[:, i] * discount
        ndcg = dcg / (ideal_dcg + 1e-8)
        return 1.0 - ndcg.mean()


class RAGLoss(BaseLoss):
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recall_loss = RetrievalRelevanceLoss()
        self.coverage_loss = CoverageLoss()
        self.ranking_loss = RankingConsistencyLoss()

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        l_recall = self.recall_loss.forward(query_emb, doc_embs, labels)
        l_coverage = self.coverage_loss.forward(query_emb, doc_embs, labels)
        l_ndcg = self.ranking_loss.forward(query_emb, doc_embs, labels)
        return self.alpha * l_recall + self.beta * l_coverage + self.gamma * l_ndcg
