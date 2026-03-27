from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.loss.base import BaseLoss


class ContrastiveControlLoss(BaseLoss):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, dim=-1)
        pos_norm = F.normalize(positives, dim=-1)
        neg_norm = F.normalize(negatives, dim=-1)
        pos_sim = (anchor_norm * pos_norm).sum(dim=-1) / self.temperature
        neg_sim = (anchor_norm * neg_norm).sum(dim=-1) / self.temperature
        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        return loss


class KnowledgeConsistencyLoss(BaseLoss):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        super().__init__()
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
            except Exception:
                self._model = "unavailable"
        return self._model

    def _compute_entailment_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        model = self._get_model()
        if model == "unavailable":
            scores = []
            for premise, hypothesis in pairs:
                common_words = set(premise.lower().split()) & set(hypothesis.lower().split())
                total_words = set(hypothesis.lower().split())
                sim = len(common_words) / max(len(total_words), 1)
                scores.append(sim)
            return scores
        scores = model.predict(pairs)
        if len(scores.shape) > 1:
            entailment_scores = scores[:, 0].tolist()
        else:
            entailment_scores = scores.tolist()
        return entailment_scores

    def compute(self, pairs: List[Tuple[str, str]]) -> float:
        scores = self._compute_entailment_score(pairs)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        loss = 1.0 - min(max(avg_score, 0.0), 1.0)
        return loss

    def forward(self, pairs: List[Tuple[str, str]]) -> torch.Tensor:
        loss_val = self.compute(pairs)
        return torch.tensor(loss_val, requires_grad=False)


class InstructionFollowingLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def regression_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, targets)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cls_loss = self.classification_loss(logits, labels)
        if scores is not None and targets is not None:
            reg_loss = self.regression_loss(scores, targets)
            return 0.6 * cls_loss + 0.4 * reg_loss
        return cls_loss


class ControlAlignmentLoss(BaseLoss):
    def __init__(
        self,
        lambda1: float = 0.4,
        lambda2: float = 0.4,
        lambda3: float = 0.2,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.contrastive_loss = ContrastiveControlLoss()
        self.nli_loss = KnowledgeConsistencyLoss()
        self.inst_loss = InstructionFollowingLoss()

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        nli_pairs: List[Tuple[str, str]],
        inst_logits: torch.Tensor,
        inst_labels: torch.Tensor,
        inst_scores: Optional[torch.Tensor] = None,
        inst_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l_contrastive = self.contrastive_loss.forward(anchor, positives, negatives)
        l_nli = self.nli_loss.forward(nli_pairs)
        l_inst = self.inst_loss.forward(inst_logits, inst_labels, inst_scores, inst_targets)
        return self.lambda1 * l_contrastive + self.lambda2 * l_nli + self.lambda3 * l_inst
