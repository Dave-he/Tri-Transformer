from typing import List, Optional
import torch
import torch.nn as nn

from eval.loss.base import BaseLoss
from eval.loss.rag_loss import RAGLoss
from eval.loss.control_alignment_loss import ControlAlignmentLoss
from eval.loss.hallucination_loss import HallucinationLoss


class TotalLoss(BaseLoss):
    def __init__(
        self,
        w1: float = 0.3,
        w2: float = 0.4,
        w3: float = 0.3,
        rag_kwargs: dict = None,
        ctrl_kwargs: dict = None,
        hall_kwargs: dict = None,
    ):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.rag_loss = RAGLoss(**(rag_kwargs or {}))
        self.ctrl_loss = ControlAlignmentLoss(**(ctrl_kwargs or {}))
        self.hall_loss = HallucinationLoss(**(hall_kwargs or {}))

    def forward(
        self,
        query_emb: torch.Tensor,
        doc_embs: torch.Tensor,
        relevance_labels: torch.Tensor,
        ctrl_anchor: torch.Tensor,
        ctrl_positives: torch.Tensor,
        ctrl_negatives: torch.Tensor,
        nli_pairs: list,
        inst_logits: torch.Tensor,
        inst_labels: torch.Tensor,
        hall_claims: list,
        hall_docs: list,
        inst_scores: Optional[torch.Tensor] = None,
        inst_targets: Optional[torch.Tensor] = None,
        confidence_scores: Optional[torch.Tensor] = None,
        gen_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_rag = self.rag_loss.forward(query_emb, doc_embs, relevance_labels)
        l_ctrl = self.ctrl_loss.forward(
            ctrl_anchor, ctrl_positives, ctrl_negatives,
            nli_pairs, inst_logits, inst_labels, inst_scores, inst_targets
        )
        l_hall = self.hall_loss.forward(
            hall_claims, hall_docs, confidence_scores, gen_labels
        )
        return self.w1 * l_rag + self.w2 * l_ctrl + self.w3 * l_hall
