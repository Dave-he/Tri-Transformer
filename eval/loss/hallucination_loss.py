from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.loss.base import BaseLoss


class FactualHallucinationLoss(BaseLoss):
    def __init__(self):
        super().__init__()

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

    def compute(self, claims: List[str], knowledge_docs: List[List[str]]) -> float:
        losses = []
        for claim, docs in zip(claims, knowledge_docs):
            support = self._compute_factual_support(claim, docs)
            loss = 1.0 - min(support, 1.0)
            losses.append(loss)
        return sum(losses) / len(losses) if losses else 0.0

    def forward(self, claims: List[str], knowledge_docs: List[List[str]]) -> torch.Tensor:
        loss_val = self.compute(claims, knowledge_docs)
        return torch.tensor(loss_val)


class SourceAttributionLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def _compute_attribution_score(self, text: str, source_chunks: List[str]) -> float:
        text_tokens = set(text.lower().split())
        best_coverage = 0.0
        for chunk in source_chunks:
            chunk_tokens = set(chunk.lower().split())
            if not text_tokens:
                continue
            coverage = len(text_tokens & chunk_tokens) / len(text_tokens)
            best_coverage = max(best_coverage, coverage)
        return best_coverage

    def compute(self, texts: List[str], source_chunks: List[List[str]]) -> float:
        losses = []
        for text, chunks in zip(texts, source_chunks):
            attribution = self._compute_attribution_score(text, chunks)
            loss = 1.0 - min(attribution, 1.0)
            losses.append(loss)
        return sum(losses) / len(losses) if losses else 0.0

    def forward(self, texts: List[str], source_chunks: List[List[str]]) -> torch.Tensor:
        loss_val = self.compute(texts, source_chunks)
        return torch.tensor(loss_val)


class AbstentionCalibrationLoss(BaseLoss):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCELoss()

    def forward(self, confidence_scores: torch.Tensor, generation_labels: torch.Tensor) -> torch.Tensor:
        should_generate = (confidence_scores >= self.threshold).float()
        loss = F.binary_cross_entropy(
            should_generate.float(),
            generation_labels.float(),
            reduction="mean"
        )
        return loss


class HallucinationLoss(BaseLoss):
    def __init__(
        self,
        mu1: float = 0.4,
        mu2: float = 0.3,
        mu3: float = 0.3,
        abstention_threshold: float = 0.5,
    ):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.fact_loss = FactualHallucinationLoss()
        self.attr_loss = SourceAttributionLoss()
        self.abstain_loss = AbstentionCalibrationLoss(threshold=abstention_threshold)

    def compute_hallucination_rate(self, outputs_or_generations, knowledge_docs: Optional[List[List[str]]] = None) -> float:
        if knowledge_docs is not None:
            generations = outputs_or_generations
            hallucination_count = 0
            for gen, docs in zip(generations, knowledge_docs):
                support = self.fact_loss._compute_factual_support(gen, docs)
                if support < 0.3:
                    hallucination_count += 1
            return hallucination_count / len(generations) if generations else 0.0
        outputs = outputs_or_generations
        hallucination_count = 0
        for output in outputs:
            answer = output.get("generated_answer", "")
            docs = output.get("retrieved_docs", [])
            support = self.fact_loss._compute_factual_support(answer, docs)
            if support < 0.3:
                hallucination_count += 1
        return hallucination_count / len(outputs) if outputs else 0.0

    def forward(
        self,
        claims: List[str],
        knowledge_docs: List[List[str]],
        confidence_scores: Optional[torch.Tensor] = None,
        generation_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_fact = self.fact_loss.forward(claims, knowledge_docs)
        l_attr = self.attr_loss.forward(claims, knowledge_docs)
        if confidence_scores is not None and generation_labels is not None:
            l_abstain = self.abstain_loss.forward(confidence_scores, generation_labels)
        else:
            l_abstain = torch.tensor(0.0)
        return self.mu1 * l_fact + self.mu2 * l_attr + self.mu3 * l_abstain
